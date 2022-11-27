from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch
import dgl

from .graph_bert_utils import GraphBertConfig, GraphBert_path
from .discriminator import Discriminator
from transformers.models.bert.modeling_bert import BertPreTrainedModel

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class GraphClassifier(nn.Module):
    def l2_norm(self, x, dim=1):
        return x / (torch.max(torch.norm(x, dim=dim, keepdim=True), self.epsilon))

    def __init__(self, params, dl):
        super().__init__()

        self.params = params
        self.attention_weights = None
        self.gnn = RGCN(params)

        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.hidden_dim, sparse=False)

        self.fc_layer = nn.Linear(2 * self.params.num_layers * self.params.hidden_dim + 2 * self.params.hidden_dim, 1)
        # self.fc_layer = nn.Linear(2 * self.params.num_layers * self.params.hidden_dim + 4 * self.params.hidden_dim, 1)

        config = GraphBertConfig(graph_size=dl.args.nodes_all, rel_size=dl.args.rel_size, edge_size=dl.edges_train['total'],
                                 in_feat=None, node_feat=None, rel_feat=None, edge_feat=None,
                                 node_dims=None, rel_dims=None, node_features=None, rel_features=None,
                                 seq_length=params.max_seq_length, neighbor_samples=params.context_hops, path_length=params.max_path_len, path_samples=params.path_samples,
                                 _sep_id=dl.args._sep_id, _cls_id=dl.args._cls_id, _pad_id=dl.args._pad_id,
                                 label_rp=dl.args.rel_size, label_lp=2, hidden_size=params.hidden_dim, edge_hidden_size=params.hidden_dim,
                                 num_hidden_layers=params.num_layers, num_attention_heads=params.num_heads,
                                 hidden_dropout_prob=params.dropout, attention_probs_dropout_prob=params.dropout,
                                 node_id2index=None, node_index2type=None
                                 )
        self.seq_model = GraphBertForPairScoring(config)
        self.proj_ctx = nn.Linear(self.params.hidden_dim * self.params.num_layers, self.params.hidden_dim)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.params.dropout)

        self.Hn = nn.Parameter(torch.FloatTensor(dl.args.nodes_all, self.params.num_clusters))
        self.Zn = nn.Parameter(torch.FloatTensor(self.params.num_clusters, self.params.hidden_dim))
        nn.init.xavier_normal_(self.Hn)
        nn.init.xavier_normal_(self.Zn)
        self.cluster_act = nn.Softmax(dim=-1)
        self.cluster_dropout = nn.Dropout(self.params.cluster_dropout)
        self.summary_act = nn.Sigmoid()
        # self.W_c = nn.Bilinear(self.params.hidden_dim, self.params.hidden_dim, self.params.hidden_dim)
        # self.W_ht = nn.Bilinear(self.params.hidden_dim, self.params.hidden_dim, self.params.hidden_dim)

        self.k = torch.tensor(self.params.num_clusters, dtype=torch.float32, device=self.params.device, requires_grad=False)

    def forward(self, data_pos, data_neg, dl, type="t"):
        h1_l_loss, h1_o_loss, h1_s_loss, logit = [], [], [], None
        tokens0, masks, segments = [], [], []
        tokens0_neg, masks_neg, segments_neg = [], [], []
        view_embeds, view_aggr, rep_rel = None, None, None

        adj, I = dl.adj, dl.I
        degrees = dl.edges_train['adj_all'].sum(0).reshape((-1, 1))
        m = degrees.sum()
        degrees = torch.tensor(degrees, dtype=torch.float32, device=self.params.device, requires_grad=False)

        ######################################## positive subgraph modeling ########################################
        ''' enclosing subgraph view learning '''
        g, _, rel_labels = data_pos
        g.ndata['h'] = self.gnn(g)
        g_out_n = mean_nodes(g, 'repr')
        bs, vt, hd = list(g_out_n.shape)
        g_out_n = g_out_n.view(bs, vt * hd)
        g_out = self.dropout(g_out_n)
        g_out = self.act(self.proj_ctx(g_out))

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
        rel_embs = self.rel_emb(rel_labels)
        h_out = self.proj_ctx(head_embs.view(-1, self.params.num_layers * self.params.hidden_dim))
        t_out = self.proj_ctx(tail_embs.view(-1, self.params.num_layers * self.params.hidden_dim))

        ''' community view learning '''
        n_indices = g.ndata['idx'][head_ids]
        self.Hn.data[n_indices] = self.cluster_act(self.cluster_dropout(torch.spmm(h_out, self.Zn.t())))
        s1 = self.summary_act(self.dropout(torch.spmm(self.Hn.data[n_indices], self.Zn)))
        n_indices = g.ndata['idx'][tail_ids]
        self.Hn.data[n_indices] = self.cluster_act(self.cluster_dropout(torch.spmm(t_out, self.Zn.t())))
        s2 = self.summary_act(self.dropout(torch.spmm(self.Hn.data[n_indices], self.Zn)))
        s = s1 * s2

        ''' metapath view learning '''
        for hd, tl in zip(head_ids, tail_ids):
            edges_btw_roots = g.edge_id(hd, tl)
            tokens0.append(g.edata['token'][edges_btw_roots].unsqueeze(0))
            masks.append(g.edata['mask'][edges_btw_roots].unsqueeze(0))
            segments.append(g.edata['segment'][edges_btw_roots].unsqueeze(0))
        tokens = torch.cat(tokens0, dim=0).to(self.params.device) # bs x ps x sq
        masks = torch.cat(masks, dim=0).to(self.params.device) # bs x ps x sq
        segments = torch.cat(segments, dim=0).to(self.params.device) # bs x ps x sq
        rep_seq, rep_rel = self.seq_model(rel_input_ids=tokens, rel_attention_mask=masks, rel_token_type_ids=segments)
        # print("positive :", tokens)

        ''' view attention-scores and aggregation '''
        view_embeds = torch.stack([g_out, rep_seq, s], dim=1)
        rel_embeds = rel_embs.unsqueeze(1) # bs x 1 x hd
        self.attention_weights = torch.sum(rel_embeds * view_embeds, dim=-1) # bs x view
        self.attention_weights = F.softmax(self.attention_weights, dim=-1) # bs x view
        tmp_attention_weights = self.attention_weights.unsqueeze(-1) # bs x view x 1
        view_aggr = torch.sum(tmp_attention_weights * view_embeds, dim=1)  # [bs, hd]

        ''' positive triplet scoring '''
        g_rep = torch.cat([head_embs.view(-1, self.params.num_layers * self.params.hidden_dim),
                           tail_embs.view(-1, self.params.num_layers * self.params.hidden_dim),
                           rel_embs, view_aggr], dim=1)
        out_pos = self.fc_layer(g_rep)

        ######################################## negative subgraph modeling ########################################
        ''' enclosing subgraph view learning '''
        g1, _, rel_labels1 = data_neg
        g1.ndata['h'] = self.gnn(g1)
        g_out1_n = mean_nodes(g1, 'repr')
        bs, vt, hd = list(g_out1_n.shape)
        g_out1_n = g_out1_n.view(bs, vt * hd)
        g_out1 = self.dropout(g_out1_n)
        g_out1 = self.act(self.proj_ctx(g_out1))

        head_ids1 = (g1.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs1 = g1.ndata['repr'][head_ids1]
        tail_ids1 = (g1.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs1 = g1.ndata['repr'][tail_ids1]
        rel_embs1 = self.rel_emb(rel_labels1)
        hn_out = self.proj_ctx(head_embs1.view(-1, self.params.num_layers * self.params.hidden_dim))
        tn_out = self.proj_ctx(tail_embs1.view(-1, self.params.num_layers * self.params.hidden_dim))

        n_indices = g1.ndata['idx'][tail_ids1]
        self.Hn.data[n_indices] = self.cluster_act(self.cluster_dropout(torch.spmm(tn_out, self.Zn.t())))
        s3 = self.summary_act(self.dropout(torch.spmm(self.Hn.data[n_indices], self.Zn)))
        sn = s1 * s3
        if type in ["t", "v"]:
            for hd, tl in zip(head_ids, tail_ids):
                edges_btw_roots = g.edge_id(hd, tl)
                tokens0_neg.append(g.edata['token_neg'][edges_btw_roots].unsqueeze(0))
                masks_neg.append(g.edata['mask_neg'][edges_btw_roots].unsqueeze(0))
                segments_neg.append(g.edata['segment_neg'][edges_btw_roots].unsqueeze(0))
        elif type == "p":
            for hd, tl in zip(head_ids, tail_ids):
                edges_btw_roots = g.edge_id(hd, tl)
                tokens0_neg.append(g.edata['token_neg_p'][edges_btw_roots].unsqueeze(0))
                masks_neg.append(g.edata['mask_neg_p'][edges_btw_roots].unsqueeze(0))
                segments_neg.append(g.edata['segment_neg_p'][edges_btw_roots].unsqueeze(0))
        elif type == "r":
            for hd, tl in zip(head_ids, tail_ids):
                edges_btw_roots = g.edge_id(hd, tl)
                tokens0_neg.append(g.edata['token_neg_r'][edges_btw_roots].unsqueeze(0))
                masks_neg.append(g.edata['mask_neg_r'][edges_btw_roots].unsqueeze(0))
                segments_neg.append(g.edata['segment_neg_r'][edges_btw_roots].unsqueeze(0))
        elif type == "2":
            for hd, tl in zip(head_ids, tail_ids):
                edges_btw_roots = g.edge_id(hd, tl)
                tokens0_neg.append(g.edata['token_neg_2'][edges_btw_roots].unsqueeze(0))
                masks_neg.append(g.edata['mask_neg_2'][edges_btw_roots].unsqueeze(0))
                segments_neg.append(g.edata['segment_neg_2'][edges_btw_roots].unsqueeze(0))
        tokens = torch.cat(tokens0_neg, dim=0).to(self.params.device) # bs x ps x sq
        masks = torch.cat(masks_neg, dim=0).to(self.params.device) # bs x ps x sq
        segments = torch.cat(segments_neg, dim=0).to(self.params.device) # bs x ps x sq
        rep_seq_neg, rep_rel_neg = self.seq_model(rel_input_ids=tokens, rel_attention_mask=masks, rel_token_type_ids=segments)
        # print("negative :", tokens)
        view_embeds1 = torch.stack([g_out1, rep_seq_neg, sn], dim=1) #TODO negative path-sampling
        view_aggr1 = torch.sum(tmp_attention_weights * view_embeds1, dim=1)  # [bs, hd]

        ''' negative triplet scoring '''
        g_rep1 = torch.cat([head_embs1.view(-1, self.params.num_layers * self.params.hidden_dim),
                            tail_embs1.view(-1, self.params.num_layers * self.params.hidden_dim),
                            rel_embs1, view_aggr1], dim=1)
        out_neg = self.fc_layer(g_rep1)

        '''' modularity maximization based community detection '''
        a = torch.spmm(self.Hn.t(), torch.spmm(adj, self.Hn))
        ca = torch.spmm(self.Hn.t(), degrees)
        cb = torch.spmm(degrees.t(), self.Hn)
        normalizer = torch.spmm(ca, cb) / 2 / m
        spectral_loss = - torch.trace(a - normalizer) / 2 / m
        h1_l_loss.append(spectral_loss)
        pairwise = torch.spmm(self.Hn.t(), self.Hn)
        orthogonality_loss = torch.norm(pairwise / torch.norm(pairwise) - I / torch.sqrt(self.k))
        h1_o_loss.append(orthogonality_loss)
        # size_loss = torch.norm(pairwise.sum(1)) / dl.nodes['total'] * torch.sqrt(self.k) - 1
        # h1_s_loss.append(size_loss)

        ''' clustering loss computation '''
        h1_l_loss = self.params.cluster_learning_coeff * torch.stack(h1_l_loss).mean()
        h1_o_loss = self.params.cluster_orthogonality_coeff * torch.stack(h1_o_loss).mean()
        # h1_s_loss = self.params.cluster_size_coeff * torch.stack(h1_s_loss).mean()

        cluster_loss = h1_l_loss + h1_o_loss # + h1_s_loss

        return out_pos, out_neg, cluster_loss, logit, self.attention_weights, tokens0, view_embeds, view_aggr, rep_rel


class GraphBertForPairScoring(BertPreTrainedModel):
    def __init__(self, config):
        super(GraphBertForPairScoring, self).__init__(config)
        self.config = config
        self.graphbert_path = GraphBert_path(config)
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        self.proj_ctx = nn.Linear(config.hidden_size * config.path_samples * (config.num_hidden_layers+1), config.hidden_size)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, src_input_ids=None, rel_input_ids=None, tgt_input_ids=None,
                src_attention_mask=None, tgt_attention_mask=None, rel_attention_mask=None,
                src_token_type_ids=None, tgt_token_type_ids=None, rel_token_type_ids=None,
                src_embeds=None, rel_embeds=None, tgt_embeds=None,
                labels=None,
                label_dict=None, *args, **kwargs):
        bs, nt, sl = list(rel_input_ids.shape) # 1024 5 7
        seq_rel, rep_rel, pooled_seq = self.encoder_path(  # bs*nt,hn
            rel_input_ids.view(bs*nt, -1),
            attention_mask=rel_attention_mask.view(bs*nt, -1),
            token_type_ids=rel_token_type_ids.view(bs*nt, -1), inputs_embeds=None, rel_flag=True)
        rep_rel = rep_rel.view(bs, nt * self.config.hidden_size)
        pooled_seq = pooled_seq.view(bs, nt * self.config.num_hidden_layers * self.config.hidden_size)
        all_rep = torch.cat([rep_rel, pooled_seq], dim=1)
        all_rep = self.dropout(all_rep)
        seq_embeddings = self.act(self.proj_ctx(all_rep))

        return seq_embeddings, rep_rel


    def encoder_path(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None, rel_flag=True):
        outputs = self.graphbert_path(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, rel_flag=rel_flag)
        # print("outputs ", len(outputs)) 3
        # print("outputs[0] ", outputs[0].shape) # outputs[0]  torch.Size([5120, 7, 64])
        # print("outputs[1] ", outputs[1].shape) # outputs[1]  torch.Size([5120, 64])
        # print("outputs[2] ", len(outputs[2]))
        # print("outputs[2] ", outputs[2][0].shape) # torch.Size([5120, 7, 64])
        seq_output = outputs[0]
        pooled_output = outputs[1]
        pooled_seq = torch.stack(outputs[2], dim=1)
        return (seq_output, pooled_output, pooled_seq)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))

    def l2_norm(self, x, dim=1):
        return x / (torch.max(torch.norm(x, dim=dim, keepdim=True), self.epsilon))