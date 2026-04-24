import torch
from torch import nn
import math
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch.nn.functional as fn
import copy


def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):  #  mean_sequence_output, cov_sequence_output, test_item_mean_emb, test_item_cov_emb
                                                                # [B, D]                 [B, D]               [V, D]              [V,D]
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True)           # [B,1]
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True)   # [V,1]
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)    # [B, V] + [B,1] + [1,V]
    #ret = torch.clamp(-2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2), min=1e-24)
    #ret = torch.sqrt(ret)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    #cov_ret = torch.clamp(-2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2), min=1e-24)
    #cov_ret = torch.sqrt(cov_ret)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)),
                                torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) \
              + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret


class DistAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(DistAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.mean_query = nn.Linear(hidden_size, self.all_head_size)
        self.cov_query = nn.Linear(hidden_size, self.all_head_size)
        self.mean_key = nn.Linear(hidden_size, self.all_head_size)
        self.cov_key = nn.Linear(hidden_size, self.all_head_size)
        self.mean_value = nn.Linear(hidden_size, self.all_head_size)
        self.cov_value = nn.Linear(hidden_size, self.all_head_size)

        self.activation = nn.ELU()

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.mean_dense = nn.Linear(hidden_size, hidden_size)
        self.cov_dense = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # self.gamma = args.kernel_param

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_mean_tensor, input_cov_tensor, attention_mask):
        mixed_mean_query_layer = self.activation(self.mean_query(input_mean_tensor))
        mixed_mean_key_layer = self.activation(self.mean_key(input_mean_tensor))
        mixed_mean_value_layer = self.activation(self.mean_value(input_mean_tensor))

        mean_query_layer = self.transpose_for_scores(mixed_mean_query_layer)
        mean_key_layer = self.transpose_for_scores(mixed_mean_key_layer)
        mean_value_layer = self.transpose_for_scores(mixed_mean_value_layer)

        mixed_cov_query_layer = self.activation(self.cov_query(input_cov_tensor))
        mixed_cov_key_layer = self.activation(self.cov_key(input_cov_tensor))
        mixed_cov_value_layer = self.activation(self.cov_value(input_cov_tensor))

        cov_query_layer = self.transpose_for_scores(mixed_cov_query_layer)
        cov_key_layer = self.transpose_for_scores(mixed_cov_key_layer)
        cov_value_layer = self.transpose_for_scores(mixed_cov_value_layer)

        mean_attention_scores = torch.matmul(mean_query_layer, mean_key_layer.transpose(-1, -2))
        mean_attention_scores = mean_attention_scores / math.sqrt(self.attention_head_size)
        mean_attention_scores = mean_attention_scores + attention_mask
        mean_attention_probs = nn.Softmax(dim=-1)(mean_attention_scores)
        mean_attention_probs = self.attn_dropout(mean_attention_probs)

        cov_attention_scores = torch.matmul(cov_query_layer, cov_key_layer.transpose(-1, -2))
        cov_attention_scores = cov_attention_scores / math.sqrt(self.attention_head_size)
        cov_attention_scores = cov_attention_scores + attention_mask
        cov_attention_probs = nn.Softmax(dim=-1)(cov_attention_scores)
        cov_attention_probs = self.attn_dropout(cov_attention_probs)

        mean_context_layer = torch.matmul(mean_attention_probs, mean_value_layer)
        cov_context_layer = torch.matmul(cov_attention_probs, cov_value_layer)
        mean_context_layer = mean_context_layer.permute(0, 2, 1, 3).contiguous()
        cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.all_head_size,)

        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)

        mean_hidden_states = self.mean_dense(mean_context_layer)
        mean_hidden_states = self.out_dropout(mean_hidden_states)
        mean_hidden_states = self.LayerNorm(mean_hidden_states + input_mean_tensor)

        cov_hidden_states = self.cov_dense(cov_context_layer)
        cov_hidden_states = self.out_dropout(cov_hidden_states)
        cov_hidden_states = self.LayerNorm(cov_hidden_states + input_cov_tensor)

        return mean_hidden_states, cov_hidden_states


class DistFeedForward(nn.Module):

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(DistFeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.intermediate_act_fn = nn.ELU()

        self.dense_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DistTransformerLayer(nn.Module):

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(DistTransformerLayer, self).__init__()
        self.attention = DistAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.mean_feed_forward = DistFeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.cov_feed_forward = DistFeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.activation_func = nn.ELU()

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask):
        mean_attention_output, cov_attention_output  = self.attention(mean_hidden_states, cov_hidden_states, attention_mask)
        mean_feedforward_output = self.mean_feed_forward(mean_attention_output)
        cov_feedforward_output = self.activation_func(self.cov_feed_forward(cov_attention_output)) + 1
        return mean_feedforward_output, cov_feedforward_output


class DistTransformerEncoder(nn.Module):

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(DistTransformerEncoder, self).__init__()
        layer = DistTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            mean_hidden_states, cov_hidden_states = layer_module(mean_hidden_states, cov_hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append([mean_hidden_states, cov_hidden_states])
        if not output_all_encoded_layers:
            all_encoder_layers.append([mean_hidden_states, cov_hidden_states])
        return all_encoder_layers


class DT4SR(SequentialRecommender):

    def __init__(self, config, dataset):
        super(DT4SR, self).__init__(config, dataset)
        # load parameters info
        self.n_layers = 1
        self.n_heads = 2
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']    # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_mean_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        self.position_mean_embeddings = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.position_cov_embeddings = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.dist_trm_encoder = DistTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

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

    def add_position_mean_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_mean_embeddings(sequence)
        position_embeddings = self.position_mean_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(sequence_emb)
        return sequence_emb

    def add_position_cov_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_cov_embeddings(sequence)
        position_embeddings = self.position_cov_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(self.dropout(sequence_emb)) + 1
        return sequence_emb

    def forward(self, item_seq, item_seq_len):
        mean_sequence_emb = self.add_position_mean_embedding(item_seq)
        cov_sequence_emb = self.add_position_cov_embedding(item_seq)
        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.dist_trm_encoder(mean_sequence_emb, cov_sequence_emb, extended_attention_mask,
                                           output_all_encoded_layers=True)
        mean_sequence_output, cov_sequence_output = trm_output[-1]
        mean_sequence_output = self.gather_indexes(mean_sequence_output, item_seq_len - 1)
        cov_sequence_output = self.gather_indexes(cov_sequence_output, item_seq_len - 1)
        return mean_sequence_output, cov_sequence_output   # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        mean_sequence_output, cov_sequence_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        # if self.loss_type == 'BPR':
        #     neg_items = interaction[self.NEG_ITEM_ID]
        #     pos_items_emb = self.item_embedding(pos_items)
        #     neg_items_emb = self.item_embedding(neg_items)
        #     pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        #     neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        #     loss = self.loss_fct(pos_score, neg_score)
        #     return loss
        # else:  # self.loss_type = 'CE'

        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.item_mean_embeddings.weight   # [ |V|, D ]
        test_item_cov_emb = elu_activation(self.item_cov_embeddings.weight) + 1
        logits = - wasserstein_distance_matmul(mean_sequence_output, cov_sequence_output, test_item_mean_emb, test_item_cov_emb)
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
        mean_sequence_output, cov_sequence_output = self.forward(item_seq, item_seq_len)

        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.item_mean_embeddings.weight  # [ |V|, D ]
        test_item_cov_emb = elu_activation(self.item_cov_embeddings.weight) + 1
        scores = - wasserstein_distance_matmul(mean_sequence_output, cov_sequence_output, test_item_mean_emb, test_item_cov_emb)
        # test_items_emb = self.item_embedding.weight               # [ |V|, D ]
        # scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B,D]*[D,|V| ] =  [B n_items]
        return scores
