import logging
import lmdb
import numpy as np
import dgl
from torch.utils.data import Dataset
from utils_graph import ssp_heterograph_to_dgl, incidence_matrix
from sampler_graphs import *

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class SubgraphDataset(Dataset):
    def __len__(self):
        return self.num_graphs_pos

    def __init__(self, args, db_path, dl, db_name_pos, db_name_neg, add_traspose_rels=False, num_neg_samples_per_link=1):
        self.args = args
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.dl = dl
        self.num_neg_samples_per_link = num_neg_samples_per_link

        self.ssp_graph = dl.adj_list_all # (n x n), inductive, containing only train-edges
        self.num_rels = len(self.ssp_graph)
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in self.ssp_graph]
            self.ssp_graph += ssp_graph_t
        self.aug_num_rels = len(self.ssp_graph)
        self.graph = ssp_heterograph_to_dgl(self.ssp_graph, dl)

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0) # Helps because it is MultiDiGraph

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos, nodes_u, n_labels_u = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
            subgraph_u = self._prepare_subgraphs(nodes_u, r_label_pos, n_labels_u)
        subgraphs_neg, subgraphs_neg_u = [], []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg, nodes_u, n_labels_u= deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                subgraphs_neg_u.append(self._prepare_subgraphs(nodes_u, r_label_pos, n_labels_u))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)
        return subgraph_pos, subgraph_u, g_label_pos, r_label_pos, subgraphs_neg, subgraphs_neg_u, g_labels_neg, r_labels_neg

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes)) #TODO
        ''' edge_attrs=['s_t', 'mp_indices', 'type', 'weight', 'token', 'mask', 'segment'] '''
        subgraph.ndata['idx'] = self.graph.ndata['idx'][nodes]
        subgraph.edata['mp_indices'] = self.graph.edata['mp_indices'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['mp_indices_neg'] = self.graph.edata['mp_indices_neg'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['s_t'] = self.graph.edata['s_t'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['s_t_n'] = self.graph.edata['s_t_n'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['s_t_n_p'] = self.graph.edata['s_t_n_p'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['s_t_n_r'] = self.graph.edata['s_t_n_r'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['s_t_n_2'] = self.graph.edata['s_t_n_2'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['weight'] = self.graph.edata['weight'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['token'] = self.graph.edata['token'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['token_neg'] = self.graph.edata['token_neg'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['token_neg_p'] = self.graph.edata['token_neg_p'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['token_neg_r'] = self.graph.edata['token_neg_r'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['token_neg_2'] = self.graph.edata['token_neg_2'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['mask'] = self.graph.edata['mask'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['mask_neg'] = self.graph.edata['mask_neg'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['mask_neg_p'] = self.graph.edata['mask_neg_p'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['mask_neg_r'] = self.graph.edata['mask_neg_r'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['mask_neg_2'] = self.graph.edata['mask_neg_2'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['segment'] = self.graph.edata['segment'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['segment_neg'] = self.graph.edata['segment_neg'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['segment_neg_p'] = self.graph.edata['segment_neg_p'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['segment_neg_r'] = self.graph.edata['segment_neg_r'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['segment_neg_2'] = self.graph.edata['segment_neg_2'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)
        #print("n_labels ", n_labels)
        subgraph = self._prepare_features(subgraph, n_labels)

        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        n_nodes = subgraph.number_of_nodes()
        #print("self.max_n_label ", self.max_n_label)
        #print("n_labels[:, 0]] ", n_labels[:, 0])
        #print("n_labels[:, 0]] ", n_labels[:, 1])
        #print("np.arange(n_nodes) ", np.arange(n_nodes))
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        try:
            label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        except:
            try:
                label_feats[np.arange(n_nodes), self.max_n_label[0] + n_labels[:, 1]] = 1
            except:
                self.max_n_label = np.array([self.args.context_hops, self.args.context_hops])
                label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)
        self.n_feat_dim = n_feats.shape[1]
        return subgraph