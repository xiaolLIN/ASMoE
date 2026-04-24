import torch
from torch import nn
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.nn.functional as F


class BinaryActivationSTE(nn.Module):
    def __init__(self, threshold=0.5):
        super(BinaryActivationSTE, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        # forward pass
        binary_x = (x >= self.threshold).float()

        # backward pass STE
        return x + (binary_x - x).detach()


class VanillaAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor, gum_out):
        # (B, K, D) -> (B, K, 1) -> [B, K]
        energy = self.projection(input_tensor)
        mask = (1.0 - gum_out) * (-100000.0)
        energy = energy + mask.detach()
        weights = torch.softmax(energy.squeeze(-1), dim=-1)  # [B,K]
        # (B, K, D) * (B, K, 1) -> (B, D)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights


class ASMoE_SAS_S1(SequentialRecommender):

    def __init__(self, config, dataset):
        super(ASMoE_SAS_S1, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']  # encoding block number of shared expert
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.select_exp_num = config['select_exp_num']

        self.config = config

        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=False,
            batch_first=True,
        )
        self.aes_mlp = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, self.select_exp_num * 2)
        )

        self.tau_s1 = self.config['tau_s1']
        self.hard = False
        self.thre = 0.5
        self.aes_ste = BinaryActivationSTE(threshold=self.thre)

        # shared expert in stage 1
        self.shared_expert = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        # selectable experts in stage 1
        self.n_layers4select_exp = 1
        self.select_experts = nn.ModuleList([TransformerEncoder(
            n_layers=self.n_layers4select_exp,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        ) for _ in range(self.select_exp_num)])

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # exper representation fusion
        self.fusion_layer = VanillaAttention(self.hidden_size, self.hidden_size)

        self.final_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.final_dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def embedding_paras_num(self):
        item_emb_para_num = np.prod(self.item_embedding.weight.size())
        posi_emb_para_num = np.prod(self.position_embedding.weight.size())
        emb_para_num = item_emb_para_num + posi_emb_para_num
        return '\n embedding layer parameter num: {}.\n'.format(str(emb_para_num))

    def __str__(self):
        return super().__str__() + self.embedding_paras_num()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)      

        extended_attention_mask = self.get_attention_mask(item_seq)

        shared_output = self.shared_expert(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        shared_out = shared_output[-1]
        shared_exp_repr = self.gather_indexes(shared_out, item_seq_len - 1)   # [B,D]    

        gru_output, _ = self.gru(input_emb)
        tempo_repr = self.gather_indexes(gru_output, item_seq_len - 1)    # [B,D]

        gumbel_input = self.aes_mlp(tempo_repr).view(-1, self.select_exp_num, 2)  # B, selectable number, 2
        if self.training:
            gumbel_output = F.gumbel_softmax(gumbel_input, tau=self.tau_s1, hard=self.hard, dim=-1)[:,:,0].unsqueeze(-1)  # [B,selectable number,1]
        else:
            gumbel_output = (gumbel_input/self.tau_s1).softmax(-1)[:,:,0].unsqueeze(-1)

        if self.training:  # obtain all the selectable repr during training
            select_outputs = []
            for i in range(self.select_exp_num):
                select_output = self.select_experts[i](input_emb, extended_attention_mask, output_all_encoded_layers=True)
                select_out = select_output[-1]  # [B,L,D]
                select_exp_repr = self.gather_indexes(select_out, item_seq_len - 1)  # [B, D]
                select_outputs.append(select_exp_repr.unsqueeze(1))
            select_exp_reprs = torch.cat(select_outputs, dim=1)  # [B, selectable number,D]

            aes_mask = self.aes_ste(gumbel_output)  #  [B, selectable num, 1]
            select_exp_reprs = (aes_mask * select_exp_reprs)
            aes_mask = aes_mask.squeeze(-1)  # [B, selectable num]
        else:  # inference
            aes_mask = self.aes_ste(gumbel_output)  #   [B, selectable number, 1]
            aes_mask = aes_mask.squeeze(-1)  # [B, selectable num]
            select_outputs = []  # list of [B,D]  [B,D] contain zero raws
            for k in range(self.select_exp_num):
                mask_col = aes_mask[:, k].bool()
                selected_input_emb = input_emb[mask_col]  # [B_k, L, D]
                att_mask = self.get_attention_mask(item_seq=item_seq[mask_col])  # [B_k, L]
                seq_out = self.select_experts[k](selected_input_emb, att_mask, output_all_encoded_layers=True)
                selected_seq_out = self.gather_indexes(seq_out[-1], item_seq_len[mask_col] - 1)  # [B_k, D]

                seq_out_full = torch.zeros((item_seq.size(0), self.hidden_size), device=shared_exp_repr.device, dtype=shared_exp_repr.dtype)  # [B,D]
                indices = torch.nonzero(mask_col, as_tuple=True)[0]
                seq_out_full[indices] = selected_seq_out
                select_outputs.append(seq_out_full.unsqueeze(1))

            select_exp_reprs = torch.cat(select_outputs, dim=1)  # [B,K,D]

        selected_fused_repr, _ = self.fusion_layer(select_exp_reprs, aes_mask.unsqueeze(-1))

        user_repr = self.final_LayerNorm(self.final_dropout(shared_exp_repr + selected_fused_repr))
        return user_repr

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)

        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
