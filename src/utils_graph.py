import statistics
import numpy as np
import random
import scipy.sparse as ssp
import torch
import networkx as nx
import dgl
import pickle

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    ''' nodes_pos, r_label_pos, g_label_pos, n_labels_pos, nodes_u, n_labels_u '''
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_labels', 'nodes_u', 'n_labels_u')
    return dict(zip(keys, data_tuple))


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def ssp_to_torch(A, device, dense=False):
    '''
    A : Sparse adjacency matrix
    '''
    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    dat = torch.FloatTensor(A.tocoo().data)
    A = torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).to(device=device)
    return A

def ssp_heterograph_to_dgl(graph, dl, n_feats=None):
    dict_t, dict_v, dict_p, dict_r, dict_2 = None, None, None, None, None
    max_ent_len = dl.args.max_seq_length - 3
    tmp_list = [dl.args._pad_id] * dl.args.max_seq_length
    list_edge_attrs=['mp_indices', 'mp_indices_neg', 'mp_indices_neg_p', 'mp_indices_neg_r', 'mp_indices_neg_2',
                's_t', 's_t_n', 's_t_n_p', 's_t_n_r', 's_t_n_2',
                'type', 'weight',
                'token', 'token_neg', 'token_neg_p', 'token_neg_r', 'token_neg_2',
                'mask', 'mask_neg', 'mask_neg_p', 'mask_neg_r', 'mask_neg_2',
                'segment', 'segment_neg', 'segment_neg_p', 'segment_neg_r', 'segment_neg_2']

    g_nx = nx.MultiDiGraph()
    for it in list(range(dl.args.nodes_all)):
        g_nx.add_node(it, idx=it)
    # for it in list(range(graph[0].shape[0])):
    #     g_nx.add_node(it, idx=it)

    nx_triplets = []

    for r_id in sorted(dl.edges_train['edges'].keys()):
        indx = 0
        for h_id in dl.train_pos[r_id][0]:
            dict_t = None
            t_id = dl.train_pos[r_id][1][indx]
            nt_id = dl.train_neg[r_id][1][indx]
            neg_triplet = (h_id, nt_id)
            indx += 1
            key = tuple([h_id, t_id])
            wt = dl.edges_train['edges'][r_id][key]

            rel_input_ids_list, rel_mask_ids_list, rel_segment_ids_list = [], [], []
            rel_input_ids_listn, rel_mask_ids_listn, rel_segment_ids_listn = [], [], []

            paths = list(dl.train_ht2paths[h_id, tuple([r_id]), t_id])
            paths_neg = list(dl.train_neg_paths[h_id, tuple([r_id]), nt_id])
            if len(paths) > 0 and len(paths_neg) > 0: # Not all the train triplets have paths between S and T or RWs capture them
                path_len = len(paths)
                paths_neg = paths_neg[:path_len]
                mp_indices, mp_indices_neg = [], []

                for item in paths:
                    mp_indices.append(dl.path2id[item])
                for i, item in enumerate(paths):
                    paths[i] = paths[i][:max_ent_len+1]
                paths = paths[:dl.args.path_samples]
                mp_indices = mp_indices[:dl.args.path_samples]

                for item in paths_neg:
                    mp_indices_neg.append(-1)#.append(dl.path2id[item])
                for i, item in enumerate(paths_neg):
                    paths_neg[i] = paths_neg[i][:max_ent_len+1]
                paths_neg = paths_neg[:dl.args.path_samples]
                mp_indices_neg = mp_indices_neg[:dl.args.path_samples]

                if len(paths) < dl.args.path_samples:
                    paths = [paths[idx%len(paths)] for idx in range(dl.args.path_samples)]
                    mp_indices = [mp_indices[idx%len(mp_indices)] for idx in range(dl.args.path_samples)]
                if len(paths_neg) < dl.args.path_samples:
                    paths_neg = [paths_neg[idx%len(paths_neg)] for idx in range(dl.args.path_samples)]
                    mp_indices_neg = [mp_indices_neg[idx%len(mp_indices_neg)] for idx in range(dl.args.path_samples)]

                r_len = [len(item) for item in paths]
                for idx, item in enumerate(paths):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_list.append(rel_input_ids)
                    rel_mask_ids_list.append(rel_mask_ids)
                    rel_segment_ids_list.append(rel_segment_ids)
                r_len = [len(item) for item in paths_neg]
                for idx, item in enumerate(paths_neg):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_listn.append(rel_input_ids)
                    rel_mask_ids_listn.append(rel_mask_ids)
                    rel_segment_ids_listn.append(rel_segment_ids)
            else:
                mp_indices, mp_indices_neg = [], []
                for idx in range(dl.args.path_samples):
                    rel_input_ids_list.append(tmp_list)
                    rel_mask_ids_list.append(tmp_list)
                    rel_segment_ids_list.append(tmp_list)
                    mp_indices.append(-1)
                    rel_input_ids_listn.append(tmp_list)
                    rel_mask_ids_listn.append(tmp_list)
                    rel_segment_ids_listn.append(tmp_list)
                    mp_indices_neg.append(-1)
            dict_t = {'mp_indices': mp_indices, 'mp_indices_neg': mp_indices_neg, 'mp_indices_neg_p': mp_indices_neg, 'mp_indices_neg_r': mp_indices_neg, 'mp_indices_neg_2': mp_indices_neg,
                         's_t': (h_id, t_id), 's_t_n': neg_triplet, 's_t_n_p': neg_triplet, 's_t_n_r': neg_triplet, 's_t_n_2': neg_triplet,
                         'type': r_id, 'weight': wt,
                         'token': rel_input_ids_list, 'token_neg': rel_input_ids_listn, 'token_neg_p': rel_input_ids_listn, 'token_neg_r': rel_input_ids_listn, 'token_neg_2': rel_input_ids_listn,
                         'mask':rel_mask_ids_list, 'mask_neg':rel_mask_ids_listn, 'mask_neg_p':rel_mask_ids_listn, 'mask_neg_r':rel_mask_ids_listn, 'mask_neg_2':rel_mask_ids_listn,
                         'segment': rel_segment_ids_list, 'segment_neg': rel_segment_ids_listn, 'segment_neg_p': rel_segment_ids_listn, 'segment_neg_r': rel_segment_ids_listn, 'segment_neg_2': rel_segment_ids_listn}
            nx_triplets.append((h_id, t_id, dict_t))

    for r_id in sorted(dl.edges_val['edges'].keys()):
        indx = 0
        for h_id in dl.valid_pos[r_id][0]:
            dict_v = None
            t_id = dl.valid_pos[r_id][1][indx]
            nt_id = dl.valid_neg[r_id][1][indx]
            neg_triplet = (h_id, nt_id) #tuple([h_id, r_id, nt_id])
            indx += 1
            key = tuple([h_id, t_id])
            wt = dl.edges_val['edges'][r_id][key]

            rel_input_ids_list, rel_mask_ids_list, rel_segment_ids_list = [], [], []
            rel_input_ids_listn, rel_mask_ids_listn, rel_segment_ids_listn = [], [], []

            paths = list(dl.val_ht2paths[h_id, tuple([r_id]), t_id])
            paths_neg = list(dl.valid_neg_paths[h_id, tuple([r_id]), nt_id])
            if len(paths) > 0 and len(paths_neg) > 0: # Not all the valid triplets have paths between S and T or RWs capture them
                path_len = len(paths)
                paths_neg = paths_neg[:path_len]
                mp_indices, mp_indices_neg = [], []

                for item in paths:
                    mp_indices.append(dl.path2id[item])
                for i, item in enumerate(paths):
                    paths[i] = paths[i][:max_ent_len+1]
                paths = paths[:dl.args.path_samples]
                mp_indices = mp_indices[:dl.args.path_samples]

                for item in paths_neg:
                    mp_indices_neg.append(-1)#mp_indices_neg.append(dl.path2id[item])
                for i, item in enumerate(paths_neg):
                    paths_neg[i] = paths_neg[i][:max_ent_len+1]
                paths_neg = paths_neg[:dl.args.path_samples]
                mp_indices_neg = mp_indices_neg[:dl.args.path_samples]

                if len(paths) < dl.args.path_samples:
                    paths = [paths[idx%len(paths)] for idx in range(dl.args.path_samples)]
                    mp_indices = [mp_indices[idx%len(mp_indices)] for idx in range(dl.args.path_samples)]
                if len(paths_neg) < dl.args.path_samples:
                    paths_neg = [paths_neg[idx%len(paths_neg)] for idx in range(dl.args.path_samples)]
                    mp_indices_neg = [mp_indices_neg[idx%len(mp_indices_neg)] for idx in range(dl.args.path_samples)]

                r_len = [len(item) for item in paths]
                for idx, item in enumerate(paths):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id, r: rel entities
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_list.append(rel_input_ids)
                    rel_mask_ids_list.append(rel_mask_ids)
                    rel_segment_ids_list.append(rel_segment_ids)
                r_len = [len(item) for item in paths_neg]
                for idx, item in enumerate(paths_neg):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_listn.append(rel_input_ids)
                    rel_mask_ids_listn.append(rel_mask_ids)
                    rel_segment_ids_listn.append(rel_segment_ids)
                # print(rel_input_ids_listn)
                # rel_input_ids_list = tuple([torch.tensor(_ids, dtype=torch.long) for _ids in rel_input_ids_list])
                # rel_mask_ids_list = tuple([torch.tensor(_ids, dtype=torch.long) for _ids in rel_mask_ids_list])
                # rel_segment_ids_list = tuple([torch.tensor(_ids, dtype=torch.long) for _ids in rel_segment_ids_list])
            else:
                mp_indices, mp_indices_neg = [], []
                for idx in range(dl.args.path_samples):
                    rel_input_ids_list.append(tmp_list)
                    rel_mask_ids_list.append(tmp_list)
                    rel_segment_ids_list.append(tmp_list)
                    mp_indices.append(-1)
                    rel_input_ids_listn.append(tmp_list)
                    rel_mask_ids_listn.append(tmp_list)
                    rel_segment_ids_listn.append(tmp_list)
                    mp_indices_neg.append(-1)
            dict_v = {'mp_indices': mp_indices, 'mp_indices_neg': mp_indices_neg, 'mp_indices_neg_p': mp_indices_neg, 'mp_indices_neg_r': mp_indices_neg, 'mp_indices_neg_2': mp_indices_neg,
                         's_t': (h_id, t_id), 's_t_n': neg_triplet, 's_t_n_p': neg_triplet, 's_t_n_r': neg_triplet, 's_t_n_2': neg_triplet,
                         'type': r_id, 'weight': wt,
                         'token': rel_input_ids_list, 'token_neg': rel_input_ids_listn, 'token_neg_p': rel_input_ids_listn, 'token_neg_r': rel_input_ids_listn, 'token_neg_2': rel_input_ids_listn,
                         'mask':rel_mask_ids_list, 'mask_neg':rel_mask_ids_listn, 'mask_neg_p':rel_mask_ids_listn, 'mask_neg_r':rel_mask_ids_listn, 'mask_neg_2':rel_mask_ids_listn,
                         'segment': rel_segment_ids_list, 'segment_neg': rel_segment_ids_listn, 'segment_neg_p': rel_segment_ids_listn, 'segment_neg_r': rel_segment_ids_listn, 'segment_neg_2': rel_segment_ids_listn}
            nx_triplets.append((h_id, t_id, dict_v))
            # nx_triplets.append((h_id, t_id, {'mp_indices': mp_indices, 'mp_indices_neg': mp_indices_neg, 's_t': (h_id, t_id), 's_t_n': neg_triplet, 'type': r_id, 'weight': wt, 'token': rel_input_ids_list,
            #                                  'mask':rel_mask_ids_list, 'segment': rel_segment_ids_list, 'token_neg': rel_input_ids_listn,
            #                                  'mask_neg':rel_mask_ids_listn, 'segment_neg': rel_segment_ids_listn}))

    indices_r, indices_2 = [], []
    for r_id in sorted(dl.edges_test['edges'].keys()):
        indx = 0
        for h_id in dl.test_pos[r_id][0]:
            dict_p, dict_r, dict_2 = None, None, None
            # len_1 = len(dl.test_pos[r_id][1])
            # len_2 = len(dl.ntest_neg[r_id][1])
            # print(len_1, len_2)
            # assert len_1 == len_2
            t_id = dl.test_pos[r_id][1][indx]
            nt_id = dl.ntest_neg[r_id][1][indx]
            neg_triplet_p = (h_id, nt_id) #tuple([h_id, nt_id])
            indx += 1
            rp_triplet = tuple([h_id, t_id, r_id])
            key = tuple([h_id, t_id])
            wt = dl.edges_test['edges'][r_id][key]

            if rp_triplet in dl.t_r_pos:
                indx_r = dl.t_r_pos.index(rp_triplet)
                indices_r.append(indx_r)
                neg_r = dl.t_r_neg[indx_r][1]
                neg_triplet_r = (h_id, neg_r)
            if rp_triplet in dl.t_2_pos:
                indx_2 = dl.t_2_pos.index(rp_triplet)
                indices_2.append(indx_2)
                neg_2 = dl.t_2_neg[indx_2][1]
                neg_triplet_2 = (h_id, neg_2)

            rel_input_ids_list, rel_mask_ids_list, rel_segment_ids_list = [], [], []
            rel_input_ids_listp, rel_mask_ids_listp, rel_segment_ids_listp = [], [], []
            paths = list(dl.test_ht2paths[h_id, tuple([r_id]), t_id])
            paths_neg = list(dl.ntest_neg_paths[h_id, tuple([r_id]), nt_id])
            if len(paths) > 0 and len(paths_neg) > 0: # Not all the valid triplets have paths between S and T or RWs capture them
                path_len = len(paths)
                paths_neg = paths_neg[:path_len]
                mp_indices, mp_indices_neg = [], []
                for item in paths:
                    mp_indices.append(dl.path2id[item])
                for i, item in enumerate(paths):
                    paths[i] = paths[i][:max_ent_len+1]
                paths = paths[:dl.args.path_samples]
                mp_indices = mp_indices[:dl.args.path_samples]
                for item in paths_neg:
                    mp_indices_neg.append(-1)#mp_indices_neg.append(dl.path2id[item])
                for i, item in enumerate(paths_neg):
                    paths_neg[i] = paths_neg[i][:max_ent_len+1]
                paths_neg = paths_neg[:dl.args.path_samples]
                mp_indices_neg = mp_indices_neg[:dl.args.path_samples]
                if len(paths) < dl.args.path_samples:
                    paths = [paths[idx%len(paths)] for idx in range(dl.args.path_samples)]
                    mp_indices = [mp_indices[idx%len(mp_indices)] for idx in range(dl.args.path_samples)]
                if len(paths_neg) < dl.args.path_samples:
                    paths_neg = [paths_neg[idx%len(paths_neg)] for idx in range(dl.args.path_samples)]
                    mp_indices_neg = [mp_indices_neg[idx%len(mp_indices_neg)] for idx in range(dl.args.path_samples)]
                r_len = [len(item) for item in paths]
                for idx, item in enumerate(paths):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id, r: rel entities
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_list.append(rel_input_ids)
                    rel_mask_ids_list.append(rel_mask_ids)
                    rel_segment_ids_list.append(rel_segment_ids)
                r_len = [len(item) for item in paths_neg]
                for idx, item in enumerate(paths_neg):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_listp.append(rel_input_ids)
                    rel_mask_ids_listp.append(rel_mask_ids)
                    rel_segment_ids_listp.append(rel_segment_ids)
            else:
                mp_indices, mp_indices_neg = [], []
                for idx in range(dl.args.path_samples):
                    rel_input_ids_list.append(tmp_list)
                    rel_mask_ids_list.append(tmp_list)
                    rel_segment_ids_list.append(tmp_list)
                    mp_indices.append(-1)
                    rel_input_ids_listp.append(tmp_list)
                    rel_mask_ids_listp.append(tmp_list)
                    rel_segment_ids_listp.append(tmp_list)
                    mp_indices_neg.append(-1)
            dict_p = {'mp_indices': mp_indices, 'mp_indices_neg': mp_indices_neg, 'mp_indices_neg_p': mp_indices_neg,
                         's_t': (h_id, t_id), 's_t_n': neg_triplet_p, 's_t_n_p': neg_triplet_p,
                         'type': r_id, 'weight': wt,
                         'token': rel_input_ids_list, 'token_neg': rel_input_ids_listp, 'token_neg_p': rel_input_ids_listp,
                         'mask':rel_mask_ids_list, 'mask_neg':rel_mask_ids_listp, 'mask_neg_p':rel_mask_ids_listp,
                         'segment': rel_segment_ids_list, 'segment_neg': rel_segment_ids_listp, 'segment_neg_p': rel_segment_ids_listp}

            rel_input_ids_list, rel_mask_ids_list, rel_segment_ids_list = [], [], []
            rel_input_ids_listr, rel_mask_ids_listr, rel_segment_ids_listr = [], [], []
            if rp_triplet in dl.t_r_pos:
                paths = list(dl.test_ht2paths[h_id, tuple([r_id]), t_id])
                paths_neg = list(dl.test_r_neg_paths[h_id, tuple([r_id]), neg_r])
                if len(paths) > 0 and len(paths_neg) > 0: # Not all the valid triplets have paths between S and T or RWs capture them
                    path_len = len(paths)
                    paths_neg = paths_neg[:path_len]
                    mp_indices, mp_indices_neg = [], []
                    for item in paths:
                        mp_indices.append(dl.path2id[item])
                    for i, item in enumerate(paths):
                        paths[i] = paths[i][:max_ent_len+1]
                    paths = paths[:dl.args.path_samples]
                    mp_indices = mp_indices[:dl.args.path_samples]
                    for item in paths_neg:
                        mp_indices_neg.append(-1)#mp_indices_neg.append(dl.path2id[item])
                    for i, item in enumerate(paths_neg):
                        paths_neg[i] = paths_neg[i][:max_ent_len+1]
                    paths_neg = paths_neg[:dl.args.path_samples]
                    mp_indices_neg = mp_indices_neg[:dl.args.path_samples]
                    if len(paths) < dl.args.path_samples:
                        paths = [paths[idx%len(paths)] for idx in range(dl.args.path_samples)]
                        mp_indices = [mp_indices[idx%len(mp_indices)] for idx in range(dl.args.path_samples)]
                    if len(paths_neg) < dl.args.path_samples:
                        paths_neg = [paths_neg[idx%len(paths_neg)] for idx in range(dl.args.path_samples)]
                        mp_indices_neg = [mp_indices_neg[idx%len(mp_indices_neg)] for idx in range(dl.args.path_samples)]
                    r_len = [len(item) for item in paths]
                    for idx, item in enumerate(paths):
                        item = list(item)
                        rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                        rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                        rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id, r: rel entities
                        assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                        rel_input_ids_list.append(rel_input_ids)
                        rel_mask_ids_list.append(rel_mask_ids)
                        rel_segment_ids_list.append(rel_segment_ids)
                    r_len = [len(item) for item in paths_neg]
                    for idx, item in enumerate(paths_neg):
                        item = list(item)
                        rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                        rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                        rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                        assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                        rel_input_ids_listr.append(rel_input_ids)
                        rel_mask_ids_listr.append(rel_mask_ids)
                        rel_segment_ids_listr.append(rel_segment_ids)
                else:
                    mp_indices, mp_indices_neg = [], []
                    for idx in range(dl.args.path_samples):
                        rel_input_ids_list.append(tmp_list)
                        rel_mask_ids_list.append(tmp_list)
                        rel_segment_ids_list.append(tmp_list)
                        mp_indices.append(-1)
                        rel_input_ids_listr.append(tmp_list)
                        rel_mask_ids_listr.append(tmp_list)
                        rel_segment_ids_listr.append(tmp_list)
                        mp_indices_neg.append(-1)
                dict_r = {'mp_indices_neg_r': mp_indices_neg,
                          's_t_n_r': neg_triplet_r,
                          'token_neg_r': rel_input_ids_listr,
                          'mask_neg_r': rel_mask_ids_listr,
                          'segment_neg_r': rel_segment_ids_listr}

            rel_input_ids_list, rel_mask_ids_list, rel_segment_ids_list = [], [], []
            rel_input_ids_list2, rel_mask_ids_list2, rel_segment_ids_list2 = [], [], []
            if rp_triplet in dl.t_2_pos:
                paths = list(dl.test_ht2paths[h_id, tuple([r_id]), t_id])
                paths_neg = list(dl.test_2hop_neg_paths[h_id, tuple([r_id]), neg_2])
                if len(paths) > 0 and len(paths_neg) > 0: # Not all the valid triplets have paths between S and T or RWs capture them
                    path_len = len(paths)
                    paths_neg = paths_neg[:path_len]
                    mp_indices, mp_indices_neg = [], []
                    for item in paths:
                        mp_indices.append(dl.path2id[item])
                    for i, item in enumerate(paths):
                        paths[i] = paths[i][:max_ent_len+1]
                    paths = paths[:dl.args.path_samples]
                    mp_indices = mp_indices[:dl.args.path_samples]
                    for item in paths_neg:
                        mp_indices_neg.append(-1)#mp_indices_neg.append(dl.path2id[item])
                    for i, item in enumerate(paths_neg):
                        paths_neg[i] = paths_neg[i][:max_ent_len+1]
                    paths_neg = paths_neg[:dl.args.path_samples]
                    mp_indices_neg = mp_indices_neg[:dl.args.path_samples]
                    if len(paths) < dl.args.path_samples:
                        paths = [paths[idx%len(paths)] for idx in range(dl.args.path_samples)]
                        mp_indices = [mp_indices[idx%len(mp_indices)] for idx in range(dl.args.path_samples)]
                    if len(paths_neg) < dl.args.path_samples:
                        paths_neg = [paths_neg[idx%len(paths_neg)] for idx in range(dl.args.path_samples)]
                        mp_indices_neg = [mp_indices_neg[idx%len(mp_indices_neg)] for idx in range(dl.args.path_samples)]
                    r_len = [len(item) for item in paths]
                    for idx, item in enumerate(paths):
                        item = list(item)
                        rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                        rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                        rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id, r: rel entities
                        assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                        rel_input_ids_list.append(rel_input_ids)
                        rel_mask_ids_list.append(rel_mask_ids)
                        rel_segment_ids_list.append(rel_segment_ids)
                    r_len = [len(item) for item in paths_neg]
                    for idx, item in enumerate(paths_neg):
                        item = list(item)
                        rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                        rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                        rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                        assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                        rel_input_ids_list2.append(rel_input_ids)
                        rel_mask_ids_list2.append(rel_mask_ids)
                        rel_segment_ids_list2.append(rel_segment_ids)
                else:
                    mp_indices, mp_indices_neg = [], []
                    for idx in range(dl.args.path_samples):
                        rel_input_ids_list.append(tmp_list)
                        rel_mask_ids_list.append(tmp_list)
                        rel_segment_ids_list.append(tmp_list)
                        mp_indices.append(-1)
                        rel_input_ids_list2.append(tmp_list)
                        rel_mask_ids_list2.append(tmp_list)
                        rel_segment_ids_list2.append(tmp_list)
                        mp_indices_neg.append(-1)
                dict_2 = {'mp_indices_neg_2': mp_indices_neg,
                          's_t_n_2': neg_triplet_2,
                          'token_neg_2': rel_input_ids_list2,
                          'mask_neg_2': rel_mask_ids_list2,
                          'segment_neg_2': rel_segment_ids_list2}

            if dict_r is not None:
                dict_p['mp_indices_neg_r'] =  dict_r['mp_indices_neg_r']
                dict_p['s_t_n_r'] =  dict_r['s_t_n_r']
                dict_p['token_neg_r'] =  dict_r['token_neg_r']
                dict_p['mask_neg_r'] =  dict_r['mask_neg_r']
                dict_p['segment_neg_r'] =  dict_r['segment_neg_r']
            else:
                dict_p['mp_indices_neg_r'] =  dict_p['mp_indices_neg_p']
                dict_p['s_t_n_r'] =  dict_p['s_t_n_p']
                dict_p['token_neg_r'] =  dict_p['token_neg_p']
                dict_p['mask_neg_r'] =  dict_p['mask_neg_p']
                dict_p['segment_neg_r'] =  dict_p['segment_neg_p']

            if dict_2 is not None:
                dict_p['mp_indices_neg_2'] =  dict_2['mp_indices_neg_2']
                dict_p['s_t_n_2'] =  dict_2['s_t_n_2']
                dict_p['token_neg_2'] =  dict_2['token_neg_2']
                dict_p['mask_neg_2'] =  dict_2['mask_neg_2']
                dict_p['segment_neg_2'] =  dict_2['segment_neg_2']
            else:
                dict_p['mp_indices_neg_2'] =  dict_p['mp_indices_neg_p']
                dict_p['s_t_n_2'] =  dict_p['s_t_n_p']
                dict_p['token_neg_2'] =  dict_p['token_neg_p']
                dict_p['mask_neg_2'] =  dict_p['mask_neg_p']
                dict_p['segment_neg_2'] =  dict_p['segment_neg_p']

            nx_triplets.append((h_id, t_id, dict_p))

    #print(len(indices_r)) #1424/ 1424
    #print(len(indices_2)) #1420/ 1424

    for indx, item in enumerate(dl.t_r_pos):
        if indx not in indices_r:
            h_id = item[0]
            t_id = item[1]
            r_id = item[2]
            nt_id = dl.t_r_neg[indx][1]
            neg_triplet = (h_id, nt_id)
            key = tuple([h_id, t_id])
            wt = dl.edges_test['edges'][r_id][key]
            rel_input_ids_list, rel_mask_ids_list, rel_segment_ids_list = [], [], []
            rel_input_ids_listn, rel_mask_ids_listn, rel_segment_ids_listn = [], [], []
            paths = list(dl.test_ht2paths[h_id, tuple([r_id]), t_id])
            paths_neg = list(dl.test_r_neg_paths[h_id, tuple([r_id]), nt_id])
            if len(paths) > 0 and len(paths_neg) > 0: # Not all the valid triplets have paths between S and T or RWs capture them
                path_len = len(paths)
                paths_neg = paths_neg[:path_len]
                mp_indices, mp_indices_neg = [], []
                for item in paths:
                    mp_indices.append(dl.path2id[item])
                for i, item in enumerate(paths):
                    paths[i] = paths[i][:max_ent_len+1]
                paths = paths[:dl.args.path_samples]
                mp_indices = mp_indices[:dl.args.path_samples]
                for item in paths_neg:
                    mp_indices_neg.append(-1)#mp_indices_neg.append(dl.path2id[item])
                for i, item in enumerate(paths_neg):
                    paths_neg[i] = paths_neg[i][:max_ent_len+1]
                paths_neg = paths_neg[:dl.args.path_samples]
                mp_indices_neg = mp_indices_neg[:dl.args.path_samples]
                if len(paths) < dl.args.path_samples:
                    paths = [paths[idx%len(paths)] for idx in range(dl.args.path_samples)]
                    mp_indices = [mp_indices[idx%len(mp_indices)] for idx in range(dl.args.path_samples)]
                if len(paths_neg) < dl.args.path_samples:
                    paths_neg = [paths_neg[idx%len(paths_neg)] for idx in range(dl.args.path_samples)]
                    mp_indices_neg = [mp_indices_neg[idx%len(mp_indices_neg)] for idx in range(dl.args.path_samples)]

                r_len = [len(item) for item in paths]
                for idx, item in enumerate(paths):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id, r: rel entities
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_list.append(rel_input_ids)
                    rel_mask_ids_list.append(rel_mask_ids)
                    rel_segment_ids_list.append(rel_segment_ids)
                r_len = [len(item) for item in paths_neg]
                for idx, item in enumerate(paths_neg):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_listn.append(rel_input_ids)
                    rel_mask_ids_listn.append(rel_mask_ids)
                    rel_segment_ids_listn.append(rel_segment_ids)
            else:
                mp_indices, mp_indices_neg = [], []
                for idx in range(dl.args.path_samples):
                    rel_input_ids_list.append(tmp_list)
                    rel_mask_ids_list.append(tmp_list)
                    rel_segment_ids_list.append(tmp_list)
                    mp_indices.append(-1)
                    rel_input_ids_listn.append(tmp_list)
                    rel_mask_ids_listn.append(tmp_list)
                    rel_segment_ids_listn.append(tmp_list)
                    mp_indices_neg.append(-1)
            dict_r = {'mp_indices': mp_indices, 'mp_indices_neg': mp_indices_neg, 'mp_indices_neg_p': mp_indices_neg, 'mp_indices_neg_r': mp_indices_neg, 'mp_indices_neg_2': mp_indices_neg,
                      's_t': (h_id, t_id), 's_t_n': neg_triplet, 's_t_n_p': neg_triplet, 's_t_n_r': neg_triplet, 's_t_n_2': neg_triplet,
                      'type': r_id, 'weight': wt,
                      'token': rel_input_ids_list, 'token_neg': rel_input_ids_listn, 'token_neg_p': rel_input_ids_listn, 'token_neg_r': rel_input_ids_listn, 'token_neg_2': rel_input_ids_listn,
                      'mask':rel_mask_ids_list, 'mask_neg':rel_mask_ids_listn, 'mask_neg_p':rel_mask_ids_listn, 'mask_neg_r':rel_mask_ids_listn, 'mask_neg_2':rel_mask_ids_listn,
                      'segment': rel_segment_ids_list, 'segment_neg': rel_segment_ids_listn, 'segment_neg_p': rel_segment_ids_listn, 'segment_neg_r': rel_segment_ids_listn, 'segment_neg_2': rel_segment_ids_listn}
            nx_triplets.append((h_id, t_id, dict_r))

    for indx, item in enumerate(dl.t_2_pos):
        if indx not in indices_2:
            h_id = item[0]
            t_id = item[1]
            r_id = item[2]
            nt_id = dl.t_2_neg[indx][1]
            neg_triplet = (h_id, nt_id)
            key = tuple([h_id, t_id])
            wt = dl.edges_test['edges'][r_id][key]
            rel_input_ids_list, rel_mask_ids_list, rel_segment_ids_list = [], [], []
            rel_input_ids_listn, rel_mask_ids_listn, rel_segment_ids_listn = [], [], []
            paths = list(dl.test_ht2paths[h_id, tuple([r_id]), t_id])
            paths_neg = list(dl.test_2hop_neg_paths[h_id, tuple([r_id]), nt_id])
            if len(paths) > 0 and len(paths_neg) > 0: # Not all the valid triplets have paths between S and T or RWs capture them
                path_len = len(paths)
                paths_neg = paths_neg[:path_len]
                mp_indices, mp_indices_neg = [], []
                for item in paths:
                    mp_indices.append(dl.path2id[item])
                for i, item in enumerate(paths):
                    paths[i] = paths[i][:max_ent_len+1]
                paths = paths[:dl.args.path_samples]
                mp_indices = mp_indices[:dl.args.path_samples]
                for item in paths_neg:
                    mp_indices_neg.append(-1)#mp_indices_neg.append(dl.path2id[item])
                for i, item in enumerate(paths_neg):
                    paths_neg[i] = paths_neg[i][:max_ent_len+1]
                paths_neg = paths_neg[:dl.args.path_samples]
                mp_indices_neg = mp_indices_neg[:dl.args.path_samples]
                if len(paths) < dl.args.path_samples:
                    paths = [paths[idx%len(paths)] for idx in range(dl.args.path_samples)]
                    mp_indices = [mp_indices[idx%len(mp_indices)] for idx in range(dl.args.path_samples)]
                if len(paths_neg) < dl.args.path_samples:
                    paths_neg = [paths_neg[idx%len(paths_neg)] for idx in range(dl.args.path_samples)]
                    mp_indices_neg = [mp_indices_neg[idx%len(mp_indices_neg)] for idx in range(dl.args.path_samples)]

                r_len = [len(item) for item in paths]
                for idx, item in enumerate(paths):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id, r: rel entities
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_list.append(rel_input_ids)
                    rel_mask_ids_list.append(rel_mask_ids)
                    rel_segment_ids_list.append(rel_segment_ids)
                r_len = [len(item) for item in paths_neg]
                for idx, item in enumerate(paths_neg):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_listn.append(rel_input_ids)
                    rel_mask_ids_listn.append(rel_mask_ids)
                    rel_segment_ids_listn.append(rel_segment_ids)
            else:
                mp_indices, mp_indices_neg = [], []
                for idx in range(dl.args.path_samples):
                    rel_input_ids_list.append(tmp_list)
                    rel_mask_ids_list.append(tmp_list)
                    rel_segment_ids_list.append(tmp_list)
                    mp_indices.append(-1)
                    rel_input_ids_listn.append(tmp_list)
                    rel_mask_ids_listn.append(tmp_list)
                    rel_segment_ids_listn.append(tmp_list)
                    mp_indices_neg.append(-1)
            dict_2 = {'mp_indices': mp_indices, 'mp_indices_neg': mp_indices_neg, 'mp_indices_neg_p': mp_indices_neg, 'mp_indices_neg_r': mp_indices_neg, 'mp_indices_neg_2': mp_indices_neg,
                      's_t': (h_id, t_id), 's_t_n': neg_triplet, 's_t_n_p': neg_triplet, 's_t_n_r': neg_triplet, 's_t_n_2': neg_triplet,
                      'type': r_id, 'weight': wt,
                      'token': rel_input_ids_list, 'token_neg': rel_input_ids_listn, 'token_neg_p': rel_input_ids_listn, 'token_neg_r': rel_input_ids_listn, 'token_neg_2': rel_input_ids_listn,
                      'mask':rel_mask_ids_list, 'mask_neg':rel_mask_ids_listn, 'mask_neg_p':rel_mask_ids_listn, 'mask_neg_r':rel_mask_ids_listn, 'mask_neg_2':rel_mask_ids_listn,
                      'segment': rel_segment_ids_list, 'segment_neg': rel_segment_ids_listn, 'segment_neg_p': rel_segment_ids_listn, 'segment_neg_r': rel_segment_ids_listn, 'segment_neg_2': rel_segment_ids_listn}
            nx_triplets.append((h_id, t_id, dict_2))

    g_nx.add_edges_from(nx_triplets)
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, node_attrs=['idx'], edge_attrs=list_edge_attrs)

    # g_dgl = dgl.from_networkx(g_nx, node_attrs=['idx'], edge_attrs=edge_attrs)
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl

def ssp_heterograph_to_dgl_orig(graph, dl, n_feats=None):
    max_ent_len = dl.args.max_seq_length - 3
    tmp_list = [dl.args._pad_id] * dl.args.max_seq_length
    g_nx = nx.MultiDiGraph()
    for it in list(range(graph[0].shape[0])):
        g_nx.add_node(it, idx=it)
    # g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # print(dl.train_ht2paths)
    nx_triplets = []

    for idx, r_id in enumerate(sorted(dl.edges_train['edges'].keys())):
        for key in dl.edges_train['edges'][r_id]:
            src = key[0]
            dst = key[1]
            wt = dl.edges_train['edges'][r_id][key]
            rel_input_ids_list, rel_mask_ids_list, rel_segment_ids_list = [], [], []
            rel_input_ids_listn, rel_mask_ids_listn, rel_segment_ids_listn = [], [], []

            paths = list(dl.train_ht2paths[src, tuple([r_id]), dst])
            paths_neg = list(dl.train_neg_paths[src, tuple([r_id]), dst])
            if len(paths) > 0 and len(paths_neg) > 0: # Not all the train triplets have paths between S and T or RWs capture them
                path_len = len(paths)
                paths_neg = paths_neg[:path_len]
                mp_indices, mp_indices_neg = [], []

                for item in paths:
                    mp_indices.append(dl.path2id[item])
                for i, item in enumerate(paths):
                    paths[i] = paths[i][:max_ent_len+1]
                paths = paths[:dl.args.path_samples]
                mp_indices = mp_indices[:dl.args.path_samples]

                for item in paths_neg:
                    mp_indices_neg.append(-1)#mp_indices_neg.append(dl.path2id[item])
                for i, item in enumerate(paths_neg):
                    paths_neg[i] = paths_neg[i][:max_ent_len+1]
                paths_neg = paths_neg[:dl.args.path_samples]
                mp_indices_neg = mp_indices_neg[:dl.args.path_samples]

                if len(paths) < dl.args.path_samples:
                    paths = [paths[idx%len(paths)] for idx in range(dl.args.path_samples)]
                    mp_indices = [mp_indices[idx%len(mp_indices)] for idx in range(dl.args.path_samples)]
                if len(paths_neg) < dl.args.path_samples:
                    paths_neg = [paths_neg[idx%len(paths_neg)] for idx in range(dl.args.path_samples)]
                    mp_indices_neg = [mp_indices_neg[idx%len(mp_indices_neg)] for idx in range(dl.args.path_samples)]

                r_len = [len(item) for item in paths]
                for idx, item in enumerate(paths):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_list.append(rel_input_ids)
                    rel_mask_ids_list.append(rel_mask_ids)
                    rel_segment_ids_list.append(rel_segment_ids)
                r_len = [len(item) for item in paths_neg]
                for idx, item in enumerate(paths_neg):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_listn.append(rel_input_ids)
                    rel_mask_ids_listn.append(rel_mask_ids)
                    rel_segment_ids_listn.append(rel_segment_ids)
                #rel_input_ids_list = tuple([torch.tensor(_ids, dtype=torch.long) for _ids in rel_input_ids_list])
                #rel_mask_ids_list = tuple([torch.tensor(_ids, dtype=torch.long) for _ids in rel_mask_ids_list])
                #rel_segment_ids_list = tuple([torch.tensor(_ids, dtype=torch.long) for _ids in rel_segment_ids_list])
            # print("rel_input_ids_list", rel_input_ids_list)
            # print("rel_mask_ids_list", rel_mask_ids_list)
            # print("rel_segment_ids_list", rel_segment_ids_list)
            # print("src", src)
            # print("weight", wt)
            else:
                mp_indices, mp_indices_neg = [], []
                for idx in range(dl.args.path_samples):
                    rel_input_ids_list.append(tmp_list)
                    rel_mask_ids_list.append(tmp_list)
                    rel_segment_ids_list.append(tmp_list)
                    mp_indices.append(-1)
                    rel_input_ids_listn.append(tmp_list)
                    rel_mask_ids_listn.append(tmp_list)
                    rel_segment_ids_listn.append(tmp_list)
                    mp_indices_neg.append(-1)

            nx_triplets.append((src, dst, {'mp_indices': mp_indices, 'mp_indices_neg': mp_indices_neg, 's_t': (src, dst), 'type': r_id, 'weight': wt, 'token': rel_input_ids_list,
                                           'mask':rel_mask_ids_list, 'segment': rel_segment_ids_list, 'token_neg': rel_input_ids_listn,
                                           'mask_neg':rel_mask_ids_listn, 'segment_neg': rel_segment_ids_listn}))

    for idx, r_id in enumerate(sorted(dl.edges_val['edges'].keys())):
        for key in dl.edges_val['edges'][r_id]:
            src = key[0]
            dst = key[1]
            wt = dl.edges_val['edges'][r_id][key]
            rel_input_ids_list, rel_mask_ids_list, rel_segment_ids_list = [], [], []
            rel_input_ids_listn, rel_mask_ids_listn, rel_segment_ids_listn = [], [], []

            paths = list(dl.val_ht2paths[src, tuple([r_id]), dst])
            paths_neg = list(dl.valid_neg_paths[src, tuple([r_id]), dst])
            if len(paths) > 0 and len(paths_neg) > 0: # Not all the valid triplets have paths between S and T or RWs capture them
                path_len = len(paths)
                paths_neg = paths_neg[:path_len]
                mp_indices, mp_indices_neg = [], []

                for item in paths:
                    mp_indices.append(dl.path2id[item])
                for i, item in enumerate(paths):
                    paths[i] = paths[i][:max_ent_len+1]
                paths = paths[:dl.args.path_samples]
                mp_indices = mp_indices[:dl.args.path_samples]

                for item in paths_neg:
                    mp_indices_neg.append(-1)#mp_indices_neg.append(dl.path2id[item])
                for i, item in enumerate(paths_neg):
                    paths_neg[i] = paths_neg[i][:max_ent_len+1]
                paths_neg = paths_neg[:dl.args.path_samples]
                mp_indices_neg = mp_indices_neg[:dl.args.path_samples]

                if len(paths) < dl.args.path_samples:
                    paths = [paths[idx%len(paths)] for idx in range(dl.args.path_samples)]
                    mp_indices = [mp_indices[idx%len(mp_indices)] for idx in range(dl.args.path_samples)]
                if len(paths_neg) < dl.args.path_samples:
                    paths_neg = [paths_neg[idx%len(paths_neg)] for idx in range(dl.args.path_samples)]
                    mp_indices_neg = [mp_indices_neg[idx%len(mp_indices_neg)] for idx in range(dl.args.path_samples)]

                r_len = [len(item) for item in paths]
                for idx, item in enumerate(paths):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id, r: rel entities
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_list.append(rel_input_ids)
                    rel_mask_ids_list.append(rel_mask_ids)
                    rel_segment_ids_list.append(rel_segment_ids)
                r_len = [len(item) for item in paths_neg]
                for idx, item in enumerate(paths_neg):
                    item = list(item)
                    rel_input_ids = [dl.args._cls_id] + item[:r_len[idx]] + [dl.args._sep_id] + [dl.args._pad_id] * (max_ent_len+1-r_len[idx])
                    rel_mask_ids = [1] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    rel_segment_ids = [3] * (r_len[idx] + 2) + [0] * (max_ent_len+1-r_len[idx]) # 0 is the pad_id
                    assert len(rel_input_ids) == len(rel_mask_ids) == len(rel_segment_ids)
                    rel_input_ids_listn.append(rel_input_ids)
                    rel_mask_ids_listn.append(rel_mask_ids)
                    rel_segment_ids_listn.append(rel_segment_ids)
                # rel_input_ids_list = tuple([torch.tensor(_ids, dtype=torch.long) for _ids in rel_input_ids_list])
                # rel_mask_ids_list = tuple([torch.tensor(_ids, dtype=torch.long) for _ids in rel_mask_ids_list])
                # rel_segment_ids_list = tuple([torch.tensor(_ids, dtype=torch.long) for _ids in rel_segment_ids_list])
            else:
                mp_indices, mp_indices_neg = [], []
                for idx in range(dl.args.path_samples):
                    rel_input_ids_list.append(tmp_list)
                    rel_mask_ids_list.append(tmp_list)
                    rel_segment_ids_list.append(tmp_list)
                    mp_indices.append(-1)
                    rel_input_ids_listn.append(tmp_list)
                    rel_mask_ids_listn.append(tmp_list)
                    rel_segment_ids_listn.append(tmp_list)
                    mp_indices_neg.append(-1)
            nx_triplets.append((src, dst, {'mp_indices': mp_indices, 'mp_indices_neg': mp_indices_neg, 's_t': (src, dst), 'type': r_id, 'weight': wt, 'token': rel_input_ids_list,
                                           'mask':rel_mask_ids_list, 'segment': rel_segment_ids_list, 'token_neg': rel_input_ids_listn,
                                           'mask_neg':rel_mask_ids_listn, 'segment_neg': rel_segment_ids_listn}))

    g_nx.add_edges_from(nx_triplets)

    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, node_attrs=['idx'], edge_attrs=['mp_indices', 'mp_indices_neg', 's_t', 'type', 'weight', 'token', 'mask', 'segment', 'token_neg', 'mask_neg', 'segment_neg'])
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)
    return g_dgl


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """
    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def collate_dgl(samples):
    '''subgraph_pos, subgraph_u, g_label_pos, r_label_pos, subgraphs_neg, subgraphs_neg_u, g_labels_neg, r_labels_neg'''
    # The input `samples` is a list of pairs
    # graphs_pos, g_labels_pos, r_labels_pos, graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*samples))
    subgraph_pos, subgraph_u, g_label_pos, r_label_pos, subgraphs_neg, subgraphs_neg_u, g_labels_neg, r_labels_neg = map(list, zip(*samples))
    batched_graph_pos = dgl.batch(subgraph_pos)
    batched_graph_pos_u = dgl.batch(subgraph_u)
    graphs_neg = [item for sublist in subgraphs_neg for item in sublist]
    graphs_neg_u = [item for sublist in subgraphs_neg_u for item in sublist]
    g_labels_neg = [item for sublist in g_labels_neg for item in sublist]
    r_labels_neg = [item for sublist in r_labels_neg for item in sublist]

    batched_graph_neg = dgl.batch(graphs_neg)
    batched_graph_neg_u = dgl.batch(graphs_neg_u)
    return (batched_graph_pos, batched_graph_pos_u, r_label_pos), g_label_pos, (batched_graph_neg, batched_graph_neg_u, r_labels_neg), g_labels_neg


def move_batch_to_device_dgl(batch, device):
    ''' subgraph_pos, subgraph_u, g_label_pos, r_label_pos, subgraphs_neg, subgraphs_neg_u, g_labels_neg, r_labels_neg '''
    # ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg) = batch
    # subgraph_pos, subgraph_u, g_label_pos, r_label_pos, subgraphs_neg, subgraphs_neg_u, g_labels_neg, r_labels_neg = batch

    ((subgraph_pos, subgraph_u, r_label_pos), g_label_pos, (subgraphs_neg, subgraphs_neg_u, r_labels_neg), g_labels_neg)  = batch

    targets_pos = torch.LongTensor(g_label_pos).to(device=device)
    r_labels_pos = torch.LongTensor(r_label_pos).to(device=device)

    targets_neg = torch.LongTensor(g_labels_neg).to(device=device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device=device)

    g_dgl_pos = send_graph_to_device(subgraph_pos, device)
    g_dgl_neg = send_graph_to_device(subgraphs_neg, device)
    g_dgl_u = send_graph_to_device(subgraph_u, device)
    g_dgl_u_neg = send_graph_to_device(subgraphs_neg_u, device)

    return ((g_dgl_pos, g_dgl_u, r_labels_pos), targets_pos, (g_dgl_neg, g_dgl_u_neg, r_labels_neg), targets_neg)


def send_graph_to_device(g, device):
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device)

    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device)
    return g

#  The following three functions are modified from networks source codes to
#  accomodate diameter and radius for dirercted graphs


def eccentricity(G):
    e = {}
    for n in G.nbunch_iter():
        length = nx.single_source_shortest_path_length(G, n)
        e[n] = max(length.values())
    return e


def radius(G):
    e = eccentricity(G)
    e = np.where(np.array(list(e.values())) > 0, list(e.values()), np.inf)
    return min(e)


def diameter(G):
    e = eccentricity(G)
    return max(e.values())