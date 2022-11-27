from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch
import dgl

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

        self.gnn = RGCN(params)

        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.hidden_dim, sparse=False)
        self.attention_weights = None

        self.fc_layer = nn.Linear(2 * self.params.num_layers * self.params.hidden_dim + 2 * self.params.hidden_dim, 1)
        # self.fc_layer = nn.Linear(2 * self.params.num_layers * self.params.hidden_dim + 4 * self.params.hidden_dim, 1)
        # self.fc_layer = nn.Linear(self.params.num_layers * self.params.hidden_dim + self.params.edge_hidden_dim, 1)

        self.proj_ctx = nn.Linear(self.params.hidden_dim * self.params.num_layers, self.params.hidden_dim)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.params.dropout)

    def forward(self, data_pos, data_neg, dl, type="t"):
        logit, cluster_loss = None, None
        tokens0, masks, segments = [], [], []
        view_embeds, view_aggr, rep_rel = None, None, None
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

        ''' positive triplet scoring '''
        g_rep = torch.cat([head_embs.view(-1, self.params.num_layers * self.params.hidden_dim),
                           tail_embs.view(-1, self.params.num_layers * self.params.hidden_dim),
                           rel_embs, g_out], dim=1)
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

        ''' negative triplet scoring '''
        g_rep1 = torch.cat([head_embs1.view(-1, self.params.num_layers * self.params.hidden_dim),
                            tail_embs1.view(-1, self.params.num_layers * self.params.hidden_dim),
                            rel_embs1, g_out1], dim=1)
        out_neg = self.fc_layer(g_rep1)

        return out_pos, out_neg, cluster_loss, logit, self.attention_weights, tokens0, view_embeds, view_aggr, rep_rel