import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers.configuration_utils import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaPooler, RobertaAttention, RobertaIntermediate, RobertaOutput

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
BertLayerNorm = torch.nn.LayerNorm

class GraphBertEmbeddings(nn.Module):
    ''' Construct the embeddings from features, wl, position and hop vectors. '''
    def __init__(self, config):
        super(GraphBertEmbeddings, self).__init__()
        self.padding_idx = config._pad_id #8+3=11

        self.rel_feature_embeddings = nn.Embedding(config.label_rp+4, config.edge_hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.edge_hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size+1, config.hidden_size)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, rel_flag=False, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]

        inputs_embeds = self.rel_feature_embeddings(input_ids)

        # if token_type_ids is None:
        #     if hasattr(self, "token_type_ids"):
        #         buffered_token_type_ids = self.token_type_ids[:, :seq_length]
        #         buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
        #         token_type_ids = buffered_token_type_ids_expanded
        #     else:
        #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # embeddings = inputs_embeds + token_type_embeddings

        embeddings = inputs_embeds

        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)   #embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class GraphBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class GraphBertEncoder(nn.Module):
    def __init__(self, config):
        super(GraphBertEncoder, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([GraphBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True, residual_h=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            # if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            #layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            hidden_states = layer_outputs[0]

            #---- add residual ----
            if residual_h is not None:
                for index in range(hidden_states.size()[1]):
                    hidden_states[:,index,:] += residual_h

            if self.output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                if self.config.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Add last layer
        # if self.output_hidden_states:
        # all_hidden_states = all_hidden_states + (hidden_states,)

        # outputs = (hidden_states,)
        # if self.output_hidden_states:
        #     outputs = outputs + (all_hidden_states,)
        # if self.output_attentions:
        #     outputs = outputs + (all_attentions,)
        # return outputs  # last-layer hidden state, (all hidden states), (all attentions)
        # if not return_dict:
        #     return tuple(
        #         v
        #         for v in [
        #             hidden_states,
        #             next_decoder_cache,
        #             all_hidden_states,
        #             all_self_attentions,
        #             all_cross_attentions,
        #         ]
        #         if v is not None
        #     )
        # return BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_decoder_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        #     cross_attentions=all_cross_attentions,
        # )
        return tuple([hidden_states, all_hidden_states])
        #     v
        #     for v in [
        #         hidden_states,
        #         next_decoder_cache,
        #         all_hidden_states,
        #         all_self_attentions,
        #         all_cross_attentions,
        #     ]
        #     if v is not None
        # )

class GraphBert_path(RobertaPreTrainedModel):
    def __init__(self, config):
        super(GraphBert_path, self).__init__(config)
        self.config = config
        self.embeddings = GraphBertEmbeddings(config)
        self.encoder = GraphBertEncoder(config)
        self.pooler = RobertaPooler(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None, residual_h=None, rel_flag=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        # if input_ids is not None and inputs_embeds is not None:
        #    pass
        #    raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length, rel_flag=rel_flag,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, residual_h=residual_h,
        )
        # print("encoder_outputs ", len(encoder_outputs)) 2: if output hidden_states are true
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        hidden_pooled_output = []
        for item in encoder_outputs[1:][0]:
            hidden_pooled_output.append(self.pooler(item))
        return (sequence_output, pooled_output, hidden_pooled_output) # encoder_outputs[1:] # (embedding_output, embedding_output) # (sequence_output, pooled_output) + encoder_outputs[1:]

class GraphBertConfig(PretrainedConfig):
    def __init__(
            self,
            residual_type = 'graph_raw',
            graph_size=None,
            rel_size = None,
            edge_size = None,
            in_feat = None,
            node_feat = None,
            rel_feat = None,
            edge_feat = None,
            node_dims = None,
            rel_dims = None,
            node_features=None,
            rel_features=None,
            seq_length = None,
            neighbor_samples = None,
            path_length = None,
            path_samples = None,
            label_rp = None,
            label_lp = None,
            _sep_id = None,
            _cls_id = None,
            _pad_id = None,
            type_vocab_size=3, # 1) source/target=1, relation=2; or 2) source/ target=1, context=2, meta-relation=3
            max_position_embeddings=512,
            hidden_size=None,
            edge_hidden_size=None,
            num_hidden_layers=None,
            num_attention_heads=None,
            intermediate_size=64,
            hidden_act="gelu",
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            node_id2index=None,
            node_index2type=None,
            **kwargs
    ):
        super(GraphBertConfig, self).__init__(**kwargs)
        self.residual_type = residual_type
        self.graph_size = graph_size
        self.rel_size = rel_size
        self.edge_size = edge_size
        self.in_feat = in_feat
        self.node_feat = node_feat
        self.rel_feat = rel_feat
        self.edge_feat = edge_feat
        self.node_dims = node_dims
        self.rel_dims = rel_dims
        self.seq_length = seq_length
        self.neighbor_samples = neighbor_samples
        self.path_length = path_length
        self.path_samples = path_samples
        self.label_rp = label_rp
        self.label_lp = label_lp
        self._sep_id = _sep_id
        self._cls_id = _cls_id
        self._pad_id = _pad_id

        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_size
        self.edge_hidden_size = edge_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.node_features = node_features
        self.rel_features = rel_features
        self.node_id2index = node_id2index
        self.node_index2type = node_index2type


