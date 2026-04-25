import torch
from torch import nn
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss
import torch.nn.functional as F


class BinaryActivationSTE(nn.Module):
    def __init__(self, threshold=0.5):
        super(BinaryActivationSTE, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        # forward pass
        binary_x = (x >= self.threshold).float()

        # # backward pass STE
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


class CategoryAwareMaskGenerator(nn.Module):

    def __init__(self, hidden_size, tau_s2, hidden_size_cat, threshold=0.5):
        super(CategoryAwareMaskGenerator, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1 + hidden_size_cat * 3 + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

        self.tau_s2 = tau_s2
        self.ste = BinaryActivationSTE(threshold=threshold)

    def forward(self, item_emb, seq_cat_emb, target_cat_emb):
        # item_emb: [B, L, D]
        # seq_cat_emb: [B, L, C]
        # target_cat_emb: [B, C] -> [B, L, C]

        B, L, _ = item_emb.size()
        target_cat_expanded = target_cat_emb.unsqueeze(1).expand(B, L, -1)
        elem_product = seq_cat_emb * target_cat_expanded  # [B, L, C]

        # 1. Similarity Score
        sim_score = F.cosine_similarity(seq_cat_emb, target_cat_expanded, dim=-1).unsqueeze(-1)  # [B, L, 1]

        # 2. concat
        input_feats = torch.cat([sim_score, seq_cat_emb, target_cat_expanded, elem_product, item_emb], dim=-1)

        logits = self.mlp(input_feats)  # [B, L, 2]
        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.tau_s2, hard=False, dim=-1)[:,:,0].unsqueeze(-1) # [B,L,1]
        else:
            probs = (logits/self.tau_s2).softmax(-1)[:,:,0].unsqueeze(-1)
        mask = self.ste(probs)  # [B, L, 1] with STE gradient
        return mask, sim_score  #  [-1, 1]


class ASMoE_SAS_S2(SequentialRecommender):

    def __init__(self, config, dataset):
        super(ASMoE_SAS_S2, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers'] # encoding block number of shared expert
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

        self.lora_rank = config['lora_rank']

        # --- stage 1 ---
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

        # enhanced with lora, shared expert in stage 2
        self.shared_expert = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            add_lora=True,
            lora_rank=self.lora_rank
        )

        # enhanced with lora, selectable experts in stage 2
        self.n_layers4select_exp = 1
        self.select_experts = nn.ModuleList([TransformerEncoder(
            n_layers=self.n_layers4select_exp,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            add_lora=True,
            lora_rank=self.lora_rank
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

        # --- stage 2 ---
        self.attribute_hidden_size = [config['hidden_size'] // 4]
        self.selected_features = config['selected_features']
        self.pooling_mode = 'mean'
        self.feature_embed_layer_list = nn.ModuleList(
            [FeatureSeqEmbLayer(dataset, self.attribute_hidden_size[0], [self.selected_features[0]], self.pooling_mode,
                                config['device'])])

        # stage flag var, initialized with 1
        self.stage = 1
        self.mask_generator = None   # to be instantiated during training
        self.reg_lmd = config['reg_lmd']

        # parameters initialization
        self.apply(self._init_weights)

        for module in self.modules():
            # the initialization of LoRA parameter A B
            if hasattr(module, 'init_lora_weights'):
                module.init_lora_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
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
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 兼容
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def process_mask_and_get_last_index(self, Mask, item_seq_len):
        device = Mask.device
        B, L = Mask.shape

        range_idx = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        valid_range = range_idx < item_seq_len.unsqueeze(1)  # [B, L]

        has_one = (Mask.long() & valid_range).any(dim=1)
        is_all_zero = ~has_one  # 找出全为 0 的行

        if is_all_zero.any():
            row_to_fix = torch.where(is_all_zero)[0]
            col_to_fix = item_seq_len[is_all_zero] - 1
            Mask[row_to_fix, col_to_fix] = 1

        valid_ones = (Mask == 1) & (range_idx < item_seq_len.unsqueeze(1))

        pos_matrix = range_idx.expand(B, L)
        masked_pos = pos_matrix.masked_fill(~valid_ones, -1)

        act_L = masked_pos.max(dim=1).values

        return Mask.float(), act_L.long() + 1

    def get_cate_emb(self, item):
        # item seq [B,L]  OR  target item [B]
        feature_table = []
        for feature_embed_layer in self.feature_embed_layer_list:
            sparse_embedding, dense_embedding = feature_embed_layer(None, item)
            sparse_embedding = sparse_embedding['item']
            dense_embedding = dense_embedding['item']
            # concat the sparse embedding and float embedding
            if sparse_embedding is not None:
                feature_table.append(sparse_embedding)
            if dense_embedding is not None:
                feature_table.append(dense_embedding)
        return feature_table[0]

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # stage 2: obtain the mask for the input seq, and the cos sim
        sim_score, item_mask = None, None
        if self.stage == 2 and self.mask_generator is not None and self.training:
            tgt_cat_emb = self.get_cate_emb(self.pos_items).squeeze(-2)  # [B,C]
            cat_seq_emb = self.get_cate_emb(item_seq).squeeze(-2)  # [B,L,C]
            item_mask, sim_score = self.mask_generator(input_emb, cat_seq_emb, tgt_cat_emb)

            # post-process the mask, to ensure that there is one item in the seq at least.
            Mask, act_len = self.process_mask_and_get_last_index(item_mask.squeeze(-1), item_seq_len)
            item_mask = Mask.unsqueeze(-1)
            item_seq_len = act_len
            input_emb = input_emb * item_mask

            # modify the attention mask
            extended_attention_mask = self.get_attention_mask(item_seq * item_mask.squeeze(-1))
        else:
            extended_attention_mask = self.get_attention_mask(item_seq)

        shared_output = self.shared_expert(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        shared_out = shared_output[-1]
        shared_exp_repr = self.gather_indexes(shared_out, item_seq_len - 1)  # [B,D]

        gru_output, _ = self.gru(input_emb)
        tempo_repr = self.gather_indexes(gru_output, item_seq_len - 1)  # [B,D]

        gumbel_input = self.aes_mlp(tempo_repr).view(-1, self.select_exp_num, 2)  # B,selectable number, 2
        if self.training:
            gumbel_output = F.gumbel_softmax(gumbel_input, tau=self.tau_s1, hard=self.hard, dim=-1)[:, :, 0].unsqueeze(-1)  # [B,selectable number,1]
        else:
            gumbel_output = (gumbel_input / self.tau_s1).softmax(-1)[:, :, 0].unsqueeze(-1)

        if self.training:  # obtain all the selectable repr during training
            select_outputs = []
            for i in range(self.select_exp_num):
                select_output = self.select_experts[i](input_emb, extended_attention_mask, output_all_encoded_layers=True)
                select_out = select_output[-1]  # [B,L,D]
                select_exp_repr = self.gather_indexes(select_out, item_seq_len - 1)  # [B, D]
                select_outputs.append(select_exp_repr.unsqueeze(1))
            select_exp_reprs = torch.cat(select_outputs, dim=1)  # [B,selectable number,D]

            aes_mask = self.aes_ste(gumbel_output)  # [B, selectable num, 1]
            select_exp_reprs = (aes_mask * select_exp_reprs)
            aes_mask = aes_mask.squeeze(-1)  # [B, selectable num]
        else:  # inference
            aes_mask = self.aes_ste(gumbel_output)
            aes_mask = aes_mask.squeeze(-1)
            select_outputs = []
            for k in range(self.select_exp_num):
                mask_col = aes_mask[:, k].bool()
                if mask_col.sum() == 0:  # Handle empty selection case
                    seq_out_full = torch.zeros((item_seq.size(0), self.hidden_size), device=shared_exp_repr.device, dtype=shared_exp_repr.dtype)
                    select_outputs.append(seq_out_full.unsqueeze(1))
                    continue

                selected_input_emb = input_emb[mask_col]  # [B_k, L, D]
                att_mask = self.get_attention_mask(item_seq=item_seq[mask_col])
                seq_out = self.select_experts[k](selected_input_emb, att_mask, output_all_encoded_layers=True)
                selected_seq_out = self.gather_indexes(seq_out[-1], item_seq_len[mask_col] - 1)

                seq_out_full = torch.zeros((item_seq.size(0), self.hidden_size), device=shared_exp_repr.device, dtype=shared_exp_repr.dtype)
                indices = torch.nonzero(mask_col, as_tuple=True)[0]
                seq_out_full[indices] = selected_seq_out
                select_outputs.append(seq_out_full.unsqueeze(1))

            select_exp_reprs = torch.cat(select_outputs, dim=1)  # [B,K,D]

        selected_fused_repr, _ = self.fusion_layer(select_exp_reprs, aes_mask.unsqueeze(-1))

        seq_output = self.final_LayerNorm(self.final_dropout(shared_exp_repr + selected_fused_repr))
        return seq_output, item_mask, sim_score

    def configure_stage2_finetuning(self, strategy='all', device=None):
        # change the flag.
        self.stage = 2

        if self.mask_generator is None:
            print("Instantiating Mask Generator for Stage 2...")
            self.mask_generator = CategoryAwareMaskGenerator(self.hidden_size, self.config['tau_s2'], self.attribute_hidden_size[0])

            self.mask_generator.apply(self._init_weights)

            if device is not None:
                self.mask_generator.to(device)
            else:
                sample_param = next(self.parameters())
                self.mask_generator.to(sample_param.device)

        print(f"Switched to Stage 2. Strategy: {strategy}")

        # 1. enable the gradient of all the parameters
        for param in self.parameters():
            param.requires_grad = True

        # 2. ensure that the mask generator of stage 2 is trainable
        for param in self.mask_generator.parameters():
            param.requires_grad = True

        # 3. freeze some parameter and set some paras trainable according to 'strategy'
        if strategy == 'all': # all paras trainable
            pass

        elif strategy == 'only_embedding':  # only embedding is trainable
            modules_to_freeze = [
                self.shared_expert, self.select_experts, self.gru,
                self.aes_mlp, self.fusion_layer,
                self.LayerNorm, self.final_LayerNorm
            ]
            for module in modules_to_freeze:
                if hasattr(self, 'fusion_layer') and module == self.fusion_layer and self.fusion_type != 'gate':
                    continue
                for param in module.parameters():
                    param.requires_grad = False
            print("Frozen: Encoders and specific layers. Active: Embeddings & MaskGenerator.")

        elif strategy == 'only_encoder':
            modules_to_freeze = [self.item_embedding, self.position_embedding]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
            print("Frozen: Embeddings. Active: Encoders & MaskGenerator.")

        elif strategy == 'emb_lora_enc':   # our design: freeze the experts and introduce LoRA; embedding is trainable.
            for encoder in self.select_experts:
                for param in encoder.parameters():
                    param.requires_grad = False
                for layer in encoder.layer:
                    for param in layer.att_lora_layer.parameters():
                        param.requires_grad = True
                    for param in layer.ffw_lora_layer.parameters():
                        param.requires_grad = True

            for param in self.shared_expert.parameters():
                param.requires_grad = False
            for layer in self.shared_expert.layer:
                for param in layer.att_lora_layer.parameters():
                    param.requires_grad = True
                for param in layer.ffw_lora_layer.parameters():
                    param.requires_grad = True

            print("Frozen: original Encoders. Lora tuning encoders. Active: Specific layers, Embeddings & MaskGenerator.")

        else:
            raise ValueError("Unknown strategy. Choose 'all', 'only_embedding', 'only_encoder' or 'emb_lora_enc'.")

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        self.pos_items = interaction[self.POS_ITEM_ID]

        seq_output, item_mask, sim_score = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, self.pos_items)

        # introduce a similarity-aware penalty mechanism in the second stage
        if self.reg_lmd > 0:
            adv_loss = torch.mean(item_mask * F.relu(sim_score.detach()))
            loss += self.reg_lmd * adv_loss

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output, _, _= self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _, _ = self.forward(item_seq, item_seq_len)

        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
