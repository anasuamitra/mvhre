import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle as pkl
import random
from collections import defaultdict, Counter
import json
from sampler_paths import sample_paths
import torch

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class hetero_data:

    def build_graph(self, raw_examples, train=False):
        # build positive graph from triplets
        subj2objs = defaultdict(lambda: defaultdict(set))
        obj2subjs = defaultdict(lambda: defaultdict(set))
        if train:
            for _raw_ex in raw_examples:
                _head, _raw_rel, _tail  = _raw_ex
                for _rel in _raw_rel:
                    subj2objs[_head][_rel].add(_tail)
                    obj2subjs[_tail][_rel].add(_head)
        else:
            for _raw_ex in raw_examples:
                _head, _rel, _tail  = _raw_ex
                subj2objs[_head][_rel].add(_tail)
                obj2subjs[_tail][_rel].add(_head)

        return subj2objs, obj2subjs

    def __init__(self, args, path_name, edge_types=[]):
        print('Processing the heterogeneous graph ...')
        self.args = args
        self.entity_2id, self.rel_2id = defaultdict(dict), defaultdict(dict)
        self.splitted = False
        self.edge_rel2type = {}
        self.adj, self.I, self.num_mp = None, None, None
        self.path2id, self.id2path, self.id2length, self.adj_list, self.adj_list_all = None, None, None, None, None
        self.train_paths, self.val_paths, self.test_paths = None, None, None
        self.train_ht2paths, self.val_ht2paths, self.test_ht2paths = defaultdict(set), defaultdict(set), defaultdict(set)

        self.test_2hop_label, self.test_r_label = dict(), dict()
        self.train_pos, self.valid_pos, self.test_pos = dict(), dict(), dict()
        self.train_neg, self.valid_neg, self.test_2hop_neg, self.test_r_neg = dict(), dict(), dict(), dict()
        self.valid_neg_paths, self.train_neg_paths, self.ntest_neg_paths = defaultdict(set), defaultdict(set), defaultdict(set)
        self.test_r_neg_paths = defaultdict(set)
        self.t_2_pos, self.t_2_neg, self.t_r_pos, self.t_r_neg, self.ntest_neg = [], [], [], [], {}

        self.edges_train, self.edges_val, self.edges_test = self.load_links_tv(args, path_name, file_name_link='link.dat', type="tv")
        # self.edges_test = self.load_links(args, path_name, file_name_link='link.dat', type="test")

        sample_paths(self, args)
        self.calculate_hin_stats()

    def calculate_hin_stats(self):
        self.splitted = True
        self.args.train_edge_types = list(self.edges_train['edges'].keys())
        self.args.test_edge_types = list(self.edges_test['edges'].keys())
        self.args.rel_size = len(list(self.rel_2id["all"]["rel_2id"].values()))
        self.args.nodes_all = len(list(self.entity_2id["all"]["entity_2id"].values()))
        self.args.nodes_tv = len(list(self.entity_2id["tv"]["entity_2id"].values()))
        self.args.nodes_train = len(list(self.entity_2id["train"]["entity_2id"].values()))
        self.args._cls_id, self.args._sep_id, self.args._pad_id = self.args.rel_size+1, self.args.rel_size+2, self.args.rel_size+3

        adj_list = [] # Used in subgraph-sampling
        for r_id in sorted(self.edges_train['edges'].keys()): # includes all relations, except the validation and test one
            adj = self.edges_train['mat'][r_id] # (tv x tv)
            adj_list.append(adj)
        self.adj_list = adj_list
        adj_list = [] # Used in subgraph-sampling
        for r_id in sorted(self.edges_train['edges'].keys()): # includes all relations, except the validation and test one
            adj = self.edges_train['mat_all'][r_id] # (n x n)
            adj_list.append(adj)
        self.adj_list_all = adj_list

        src, tgt = self.edges_train['adj_all'].nonzero()
        data = np.squeeze(np.asarray(self.edges_train['adj_all'][src,tgt]))
        row, col, d = list(src), list(tgt), list(data)
        indices = np.vstack((row, col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(d)
        shape = (self.args.nodes_all, self.args.nodes_all) # self.edges_train['adj'].shape # (self.args.nodes_all, self.args.nodes_all)
        self.adj = torch.sparse.FloatTensor(i, v, torch.Size(shape)).requires_grad_(False).to(self.args.device)

        x = sp.eye(self.args.num_clusters)
        src, tgt = x.nonzero()
        data = np.array([1.0]*src.shape[0])
        row, col, d = list(src), list(tgt), list(data)
        indices = np.vstack((row, col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(d)
        shape = x.shape
        self.I = torch.sparse.FloatTensor(i, v, torch.Size(shape)).requires_grad_(False).to(self.args.device)

        ''' Used for negative sampling during training '''
        raw_paths = list()
        #print(self.train_ht2paths)
        for key, value in self.train_ht2paths.items():
            tmp = self.train_ht2paths[key]
            head = list(key)[0]
            rel = list(key)[1]
            tail = list(key)[2]
            tmp_list = list(tmp) + list(tuple([rel]))
            raw_paths.append(tuple([head, tmp_list, tail])) # list_of_all_strings
        # for (head, relation, tail) in self.val_ht2paths:
        #    raw_paths.append(tuple([head, list(self.val_ht2paths[tuple([head, relation, tail])]) + list(tuple([relation])), tail])) # list_of_all_strings
        self.subj2objs, self.obj2subjs = self.build_graph(raw_paths, train=True)
        '''10949: defaultdict(<class 'set'>, {(4,): {8216}, (4, 1, 4): {8216}, (4, 1, 4, 1, 4): {8216}}), 1710: defaultdict(<class 'set'>, {(0,): {5609}, (0, 3, 0): {5609}, (0, 3, 0, 3, 0): {5609}}), 1931: defaultdict(<class 'set'>, {(0,): {7477}, (0, 3, 0): {7477}, (0, 3, 0, 3, 0): {7477}})})'''

        ''' Positive positions of the input triplets '''
        for r_id in sorted(self.edges_train['edges'].keys()):
            self.train_pos[r_id] = [[], []]
            for (h_id, t_id) in self.edges_train['edges'][r_id]:
                self.train_pos[r_id][0].append(h_id)
                self.train_pos[r_id][1].append(t_id)
        for r_id in sorted(self.edges_val['edges'].keys()):
            self.valid_pos[r_id] = [[], []]
            for (h_id, t_id) in self.edges_val['edges'][r_id]:
                self.valid_pos[r_id][0].append(h_id)
                self.valid_pos[r_id][1].append(t_id)
        for r_id in sorted(self.edges_test['edges'].keys()):
            self.test_pos[r_id] = [[], []]
            for (h_id, t_id) in self.edges_test['edges'][r_id]:
                self.test_pos[r_id][0].append(h_id)
                self.test_pos[r_id][1].append(t_id)

        self.valid_neg, self.valid_neg_paths = self.get_valid_neg()
        self.train_neg, self.train_neg_paths = self.get_train_neg()
        self.ntest_neg, self.ntest_neg_paths = self.get_test_neg()

        test_random_neg = self.get_test_neigh_w_random()
        self.t_r_pos = test_random_neg['t_r_pos']
        self.t_r_neg = test_random_neg['t_r_neg']
        self.test_r_neg = test_random_neg['test_neigh']
        self.test_r_label = test_random_neg['test_label']
        self.test_r_neg_paths = test_random_neg['test_r_neg_paths']
        assert len(self.t_r_pos) == len(self.t_r_neg)

        test_2hop_neg = self.get_test_neigh_2hop()
        self.t_2_pos = test_2hop_neg['t_2_pos']
        self.t_2_neg = test_2hop_neg['t_2_neg']
        self.test_2hop_neg = test_2hop_neg['test_neigh']
        self.test_2hop_label = test_2hop_neg['test_label']
        self.test_2hop_neg_paths = test_2hop_neg['test_2hop_neg_paths']
        assert len(self.t_2_pos) == len(self.t_2_neg)

    def get_test_neigh_2hop(self):
        return self.get_test_neigh()

    def get_test_neigh(self):
        file_name = os.path.join(self.args.output_dir, self.args.dataset, 'test_2hop_neg'+".pkl")
        if os.path.exists(file_name):
            with open(file_name, 'rb') as handle:
                test_2hop_neg = pkl.load(handle)
            return test_2hop_neg
        else:
            random.seed(0)
            test_2hop_neg_paths = defaultdict(set)
            neg_neigh, pos_neigh, test_neigh, test_label = dict(), dict(), dict(), dict()
            edge_types = self.edges_test['edges'].keys()
            '''get sec_neigh'''
            pos_links = 0
            for r_id in sorted(self.edges_train['mat_all'].keys()):
                pos_links += self.edges_train['mat_all'][r_id] + self.edges_train['mat_all'][r_id].T
            for r_id in sorted(self.edges_val['mat'].keys()):
                pos_links += self.edges_val['mat'][r_id] + self.edges_val['mat'][r_id].T
            for r_id in sorted(self.edges_test['mat'].keys()):
                pos_links += self.edges_test['mat'][r_id] + self.edges_test['mat'][r_id].T
            # print("pos_links ", pos_links) # coo_matrix of train+valid+test links, weights > 1
            if self.args.dataset not in ['freebase', 'pubmed', 'acm', 'ddb', 'dblp', 'imdb', 'nell995']:
                r_double_neighs = np.dot(pos_links, pos_links) # multiplication of adjacency matrix
            else:
                r_double_neighs = []
                for idx in range(np.shape(pos_links)[0]):
                    r_double_neighs.append(sp.coo_matrix(pos_links[idx].dot(pos_links)))
                # r_double_neighs = sp.coo_matrix(np.stack(r_double_neighs))
                r_double_neighs = sp.vstack(r_double_neighs)
            data = r_double_neighs.data # data  [56. 39. 51. ...  2.  4.  4.], weights > 1
            data[:] = 1 # data  [1. 1. 1. ... 1. 1. 1.]
            r_double_neighs = \
                sp.coo_matrix((data, r_double_neighs.nonzero()), shape=np.shape(pos_links), dtype=int) \
                - sp.coo_matrix(pos_links, dtype=int) \
                - sp.lil_matrix(np.eye(np.shape(pos_links)[0], dtype=int)) # D - A -I
            # print("r_double_neighs ", r_double_neighs.shape) # (10942, 10942)
            data = r_double_neighs.data
            pos_count_index = np.where(data > 0)
            row, col = r_double_neighs.nonzero()
            r_double_neighs = sp.coo_matrix((data[pos_count_index], (row[pos_count_index], col[pos_count_index])),
                                            shape=np.shape(pos_links))
            # print("r_double_neighs ", r_double_neighs) # all weights are 1s
            row, col = r_double_neighs.nonzero()
            data = r_double_neighs.data
            sec_index = np.where(data > 0)
            row, col = row[sec_index], col[sec_index]
            # Selecting negative samples, tail ids belonging to same r_ids but they are 2-hop neighbors
            relation_range = [] #[self.nodes['shift'][k] for k in range(len(self.nodes['shift']))] + [self.nodes['total']]
            # print(self.nodes['count']) # Counter({1: 5959, 0: 3025, 3: 1902, 2: 56}), Counter({1: 9203})
            # print(self.nodes['shift']) # {0: 0, 1: 3025, 2: 8984, 3: 9040}, {0: 0}
            # print(relation_range) # [0, 3025, 8984, 9040, 10942] # start -> end ids, [0, 9203]
            for r_id in self.edges_test['edges'].keys():
                neg_neigh[r_id] = defaultdict(list)
                h_type, t_type = 1, 1
                if h_type is None and t_type is None:
                    h_type, t_type = 0, 0
                r_id_index = np.where((row >= 0) & (row < self.args.nodes_all)
                                      & (col >= 0) & (col < self.args.nodes_all))[0]
                # print("relation_range[h_type] ", relation_range[h_type])
                # print(r_id) # 6, 0
                # print(r_id_index) # [   3040    3041    3042 ... 9638395 9638396 9638397]
                # r_num = np.zeros((3, 3))
                # for h_id, t_id in zip(row, col):
                #     r_num[self.get_node_type(h_id)][self.get_node_type(t_id)] += 1
                r_row, r_col = row[r_id_index], col[r_id_index]
                for h_id, t_id in zip(r_row, r_col):
                    neg_neigh[r_id][h_id].append(t_id)

            for r_id in sorted(edge_types):
                '''get pos_neigh'''
                pos_neigh[r_id] = defaultdict(list)
                (row, col), data = self.edges_test['mat'][r_id].nonzero(), self.edges_test['mat'][r_id].data
                for h_id, t_id in zip(row, col):
                    pos_neigh[r_id][h_id].append(t_id)

                '''sample neg as same number as pos for each head node'''
                test_neigh[r_id] = [[], []]
                pos_list = [[], []]
                test_label[r_id] = []
                for h_id in sorted(list(pos_neigh[r_id].keys())):
                    pos_list[0] = [h_id] * len(pos_neigh[r_id][h_id])
                    pos_list[1] = pos_neigh[r_id][h_id]
                    test_neigh[r_id][0].extend(pos_list[0])
                    test_neigh[r_id][1].extend(pos_list[1])
                    test_label[r_id].extend([1] * len(pos_list[0]))

                    neg_list = random.choices(neg_neigh[r_id][h_id], k=len(pos_list[0])) if len(
                        neg_neigh[r_id][h_id]) != 0 else []
                    test_neigh[r_id][0].extend([h_id] * len(neg_list))
                    test_neigh[r_id][1].extend(neg_list)
                    test_label[r_id].extend([0] * len(neg_list))

                    if len(neg_list)>0:
                        for ih, it in zip(pos_list[0], pos_list[1]):
                            neg_paths = set()
                            neg_elem = random.choice(neg_list)
                            self.t_2_pos.append((ih, it, r_id))
                            self.t_2_neg.append((ih, neg_elem, r_id))
                            r_paths = self.test_ht2paths[tuple([ih, tuple([r_id]), it])] if tuple([ih, tuple([r_id]), it]) in self.test_ht2paths else None
                            h_path = self.subj2objs[ih].keys()
                            tn_path = self.obj2subjs[neg_elem].keys()
                            neg_paths = neg_paths.union(h_path)
                            neg_paths = neg_paths.union(tn_path)
                            if r_paths is not None:
                                neg_paths = neg_paths - r_paths
                                for r in r_paths:
                                    elems_discard = []
                                    for it in neg_paths:
                                        if set(r).issubset(set(it)):
                                            elems_discard.append(it)
                                    for it in elems_discard:
                                        neg_paths.remove(it)
                            neg_paths = list(neg_paths)
                            random.shuffle(neg_paths)
                            neg_paths = set(neg_paths) if len(neg_paths) >= 1 else None
                            if neg_paths is not None:
                                test_2hop_neg_paths[tuple([ih, tuple([r_id]), neg_elem])] = neg_paths

            test_2hop_neg = {}
            test_2hop_neg['t_2_pos'] = self.t_2_pos
            test_2hop_neg['t_2_neg'] = self.t_2_neg
            test_2hop_neg['test_neigh'] = test_neigh
            test_2hop_neg['test_label'] = test_label
            test_2hop_neg['test_2hop_neg_paths'] = test_2hop_neg_paths
            with open(file_name, 'wb') as handle:
                pkl.dump(test_2hop_neg, handle, protocol=pkl.HIGHEST_PROTOCOL)

            return test_2hop_neg

    def get_test_neigh_w_random(self):
        file_name = os.path.join(self.args.output_dir, self.args.dataset, 'test_random_neg'+".pkl")
        if os.path.exists(file_name):
            with open(file_name, 'rb') as handle:
                test_random_neg = pkl.load(handle)
            return test_random_neg
        else:
            random.seed(0)
            test_r_neg_paths = defaultdict(set)
            all_had_neigh = defaultdict(list)
            neg_neigh, pos_neigh, test_neigh, test_label = dict(), dict(), dict(), dict()
            edge_types = self.edges_test['edges'].keys()
            '''get pos_links of train and test data'''
            pos_links = 0
            for r_id in sorted(self.edges_train['mat_all'].keys()):
                pos_links += self.edges_train['mat_all'][r_id] + self.edges_train['mat_all'][r_id].T
            for r_id in sorted(self.edges_val['mat'].keys()):
                pos_links += self.edges_val['mat'][r_id] + self.edges_val['mat'][r_id].T
            for r_id in sorted(self.edges_test['mat'].keys()):
               pos_links += self.edges_test['mat'][r_id] + self.edges_test['mat'][r_id].T

            row, col = pos_links.nonzero()
            for h_id, t_id in zip(row, col):
                all_had_neigh[h_id].append(t_id)
            for h_id in all_had_neigh.keys():
                all_had_neigh[h_id] = set(all_had_neigh[h_id])
            for r_id in edge_types:
                t_range = range(0, self.args.nodes_all)
                pos_neigh[r_id], neg_neigh[r_id] = defaultdict(list), defaultdict(list)
                (row, col), data = self.edges_test['mat'][r_id].nonzero(), self.edges_test['mat'][r_id].data
                for h_id, t_id in zip(row, col):
                    pos_neigh[r_id][h_id].append(t_id)
                    # neg_t = random.randrange(t_range[0], t_range[1])
                    neg_t = list(t_range) #random.randrange(t_range[0], t_range[1])
                    neg_elem = random.choice(neg_t)
                    while neg_elem in all_had_neigh[h_id]:
                        neg_elem = random.choice(neg_t)
                        # neg_t = random.randrange(t_range[0], t_range[1]) # random shuffling of tail indices
                    neg_neigh[r_id][h_id].append(neg_elem)
                '''get the test_neigh'''
                test_neigh[r_id] = [[], []]
                pos_list = [[], []]
                neg_list = [[], []]
                test_label[r_id] = []
                for h_id in sorted(list(pos_neigh[r_id].keys())):
                    pos_list[0] = [h_id] * len(pos_neigh[r_id][h_id])
                    pos_list[1] = pos_neigh[r_id][h_id]
                    test_neigh[r_id][0].extend(pos_list[0])
                    test_neigh[r_id][1].extend(pos_list[1])
                    test_label[r_id].extend([1] * len(pos_neigh[r_id][h_id]))
                    neg_list[0] = [h_id] * len(neg_neigh[r_id][h_id])
                    neg_list[1] = neg_neigh[r_id][h_id]
                    test_neigh[r_id][0].extend(neg_list[0])
                    test_neigh[r_id][1].extend(neg_list[1])
                    test_label[r_id].extend([0] * len(neg_neigh[r_id][h_id]))

                    if len(neg_list[1])>0:
                        for ih, it in zip(pos_list[0], pos_list[1]):
                            neg_paths = set()
                            neg_elem = random.choice(neg_list[1])
                            # for i_nt in neg_list[1]:
                            self.t_r_pos.append((ih, it, r_id))
                            self.t_r_neg.append((ih, neg_elem, r_id))
                            r_paths = self.test_ht2paths[tuple([ih, tuple([r_id]), it])] if tuple([ih, tuple([r_id]), it]) in self.test_ht2paths else None
                            h_path = self.subj2objs[ih].keys()
                            tn_path = self.obj2subjs[neg_elem].keys()
                            neg_paths = neg_paths.union(h_path)
                            neg_paths = neg_paths.union(tn_path)
                            if r_paths is not None:
                                neg_paths = neg_paths - r_paths
                                for r in r_paths:
                                    elems_discard = []
                                    for it in neg_paths:
                                        if set(r).issubset(set(it)):
                                            elems_discard.append(it)
                                    for it in elems_discard:
                                        neg_paths.remove(it)
                            neg_paths = list(neg_paths)
                            random.shuffle(neg_paths)
                            neg_paths = set(neg_paths) if len(neg_paths) >= 1 else None
                            if neg_paths is not None:
                                test_r_neg_paths[tuple([ih, tuple([r_id]), neg_elem])] = neg_paths

            test_random_neg = {}
            test_random_neg['t_r_pos'] = self.t_r_pos
            test_random_neg['t_r_neg'] = self.t_r_neg
            test_random_neg['test_neigh'] = test_neigh
            test_random_neg['test_label'] = test_label
            test_random_neg['test_r_neg_paths'] = test_r_neg_paths
            with open(file_name, 'wb') as handle:
                pkl.dump(test_random_neg, handle, protocol=pkl.HIGHEST_PROTOCOL)

            return test_random_neg

    def get_test_neg(self, edge_types=None):
        file_name1 = os.path.join(self.args.output_dir, self.args.dataset, 'ntest_neg'+".pkl")
        file_name2 = os.path.join(self.args.output_dir, self.args.dataset, 'ntest_neg_paths'+".pkl")
        if os.path.exists(file_name1) and os.path.exists(file_name2):
            with open(file_name1, 'rb') as handle:
                ntest_neg = pkl.load(handle)
            with open(file_name2, 'rb') as handle:
                ntest_neg_paths = pkl.load(handle)
            return ntest_neg, ntest_neg_paths
        else:
            edge_types = list(self.edges_test['edges'].keys()) if edge_types is None else edge_types
            ntest_neg = dict()
            ntest_neg_paths = defaultdict(set)
            for r_id in sorted(edge_types):
                t_range = range(0, self.args.nodes_all)
                ntest_neg[r_id] = [[], []]
                ntest_neg[r_id][0] = [-1] * len(self.test_pos[r_id][0])
                ntest_neg[r_id][1] = [-1] * len(self.test_pos[r_id][1])
                idx = 0
                for h_id in self.test_pos[r_id][0]:
                    neg_paths = set()
                    neg_elem = -1
                    ntest_neg[r_id][0][idx] = h_id # Corrupt only tails, path-aware corruption
                    t_id = self.test_pos[r_id][1][idx]
                    neg_t = list(t_range) # random.randrange(t_range[0], t_range[1])
                    r_paths = self.test_ht2paths[tuple([h_id, tuple([r_id]), t_id])] if tuple([h_id, tuple([r_id]), t_id]) in self.test_ht2paths else None
                    pos_triplet = tuple([h_id, r_id, t_id])
                    pos_ent_set = set()
                    if r_paths is not None:
                        r_paths = r_paths.union(set([tuple([r_id])]))
                        for rel in r_paths:
                            pos_ent_set = pos_ent_set.union(self.subj2objs[h_id][rel])
                        neg_elem = None
                        max_iter = 10
                        while max_iter > 0:
                            neg_elem = random.choice(neg_t)
                            if (neg_elem not in pos_ent_set) : # and (neg_elem in self.obj2subjs and rel in self.obj2subjs[neg_elem] and len(self.obj2subjs[neg_elem][rel])>0 for rel in r_paths):
                                break
                            max_iter -= 1
                        if max_iter == 0:
                            neg_elem = random.choice(neg_t)
                    else:
                        neg_elem = random.choice(neg_t)
                    ntest_neg[r_id][1][idx] = neg_elem
                    # ntest_neg[r_id][1].append(neg_elem) # no check for accidentally selecting the same tail entity??
                    neg_triplet = tuple([h_id, tuple([r_id]), neg_elem])
                    # neg_path = self.val_ht2paths(tuple([h_id, r_id, t_id]))
                    h_path = self.subj2objs[h_id].keys()
                    tn_path = self.obj2subjs[neg_elem].keys()
                    neg_paths = neg_paths.union(h_path)
                    neg_paths = neg_paths.union(tn_path)
                    if r_paths is not None:
                        neg_paths = neg_paths - r_paths
                        for r in r_paths:
                            elems_discard = []
                            for it in neg_paths:
                                if set(r).issubset(set(it)):
                                    elems_discard.append(it)
                            for it in elems_discard:
                                neg_paths.remove(it)
                    neg_paths = list(neg_paths)
                    random.shuffle(neg_paths)
                    neg_paths = set(neg_paths) if len(neg_paths) >= 1 else None
                    if neg_paths is not None:
                        ntest_neg_paths[neg_triplet] = neg_paths
                    # print(pos_triplet, r_paths)
                    # print(neg_triplet, neg_paths)
                    idx += 1

            with open(file_name1, 'wb') as handle:
                pkl.dump(ntest_neg, handle, protocol=pkl.HIGHEST_PROTOCOL)
            with open(file_name2, 'wb') as handle:
                pkl.dump(ntest_neg_paths, handle, protocol=pkl.HIGHEST_PROTOCOL)
            return ntest_neg, ntest_neg_paths

    def get_train_neg(self, edge_types=None):
        file_name1 = os.path.join(self.args.output_dir, self.args.dataset, 'train_neg'+".pkl")
        file_name2 = os.path.join(self.args.output_dir, self.args.dataset, 'train_neg_paths'+".pkl")
        if os.path.exists(file_name1) and os.path.exists(file_name2):
            with open(file_name1, 'rb') as handle:
                train_neg = pkl.load(handle)
            with open(file_name2, 'rb') as handle:
                train_neg_paths = pkl.load(handle)
            return train_neg, train_neg_paths
        else:
            edge_types = list(self.edges_train['edges'].keys()) if edge_types is None else edge_types
            train_neg = dict()
            train_neg_paths = defaultdict(set)
            for r_id in sorted(edge_types):
                t_range = range(0, self.args.nodes_all)
                train_neg[r_id] = [[], []]
                train_neg[r_id][0] = [-1] * len(self.train_pos[r_id][0])
                train_neg[r_id][1] = [-1] * len(self.train_pos[r_id][1])
                idx = 0
                for h_id in self.train_pos[r_id][0]:
                    neg_paths = set()
                    neg_elem = -1
                    train_neg[r_id][0][idx] = h_id # train_neg[r_id][0].append(h_id) # Corrupt only tails, path-aware corruption
                    t_id = self.train_pos[r_id][1][idx]
                    neg_t = list(t_range) # random.randrange(t_range[0], t_range[1])
                    r_paths = self.train_ht2paths[tuple([h_id, tuple([r_id]), t_id])] if tuple([h_id, tuple([r_id]), t_id]) in self.train_ht2paths else None
                    pos_triplet = tuple([h_id, r_id, t_id])
                    pos_ent_set = set()
                    if r_paths is not None:
                        r_paths = r_paths.union(set([tuple([r_id])]))
                        for rel in r_paths:
                            pos_ent_set = pos_ent_set.union(self.subj2objs[h_id][rel])
                        neg_elem = None
                        max_iter = 10
                        while max_iter > 0:
                            neg_elem = random.choice(neg_t)
                            if (neg_elem not in pos_ent_set) : # and (neg_elem in self.obj2subjs and rel in self.obj2subjs[neg_elem] and len(self.obj2subjs[neg_elem][rel])>0 for rel in r_paths):
                                break
                            max_iter -= 1
                        if max_iter == 0:
                            neg_elem = random.choice(neg_t)
                    else:
                        neg_elem = random.choice(neg_t)
                    train_neg[r_id][1][idx] = neg_elem # train_neg[r_id][1].append(neg_elem) # no check for accidentally selecting the same tail entity??
                    neg_triplet = tuple([h_id, tuple([r_id]), neg_elem])
                    # neg_path = self.val_ht2paths(tuple([h_id, r_id, t_id]))
                    h_path = self.subj2objs[h_id].keys()
                    tn_path = self.obj2subjs[neg_elem].keys()
                    neg_paths = neg_paths.union(h_path)
                    neg_paths = neg_paths.union(tn_path)
                    if r_paths is not None:
                        neg_paths = neg_paths - r_paths
                        for r in r_paths:
                            elems_discard = []
                            for it in neg_paths:
                                if set(r).issubset(set(it)):
                                    elems_discard.append(it)
                            for it in elems_discard:
                                neg_paths.remove(it)
                    neg_paths = list(neg_paths)
                    random.shuffle(neg_paths)
                    neg_paths = set(neg_paths) if len(neg_paths) >= 1 else None
                    if neg_paths is not None:
                        train_neg_paths[neg_triplet] = neg_paths
                    idx += 1

            with open(file_name1, 'wb') as handle:
                pkl.dump(train_neg, handle, protocol=pkl.HIGHEST_PROTOCOL)
            with open(file_name2, 'wb') as handle:
                pkl.dump(train_neg_paths, handle, protocol=pkl.HIGHEST_PROTOCOL)
            return train_neg, train_neg_paths

    def get_valid_neg(self, edge_types=None):
        file_name1 = os.path.join(self.args.output_dir, self.args.dataset, 'valid_neg'+".pkl")
        file_name2 = os.path.join(self.args.output_dir, self.args.dataset, 'valid_neg_paths'+".pkl")
        if os.path.exists(file_name1) and os.path.exists(file_name2):
            with open(file_name1, 'rb') as handle:
                valid_neg = pkl.load(handle)
            with open(file_name2, 'rb') as handle:
                valid_neg_paths = pkl.load(handle)
            return valid_neg, valid_neg_paths
        else:
            edge_types = list(self.edges_val['edges'].keys()) if edge_types is None else edge_types
            valid_neg = dict()
            valid_neg_paths = defaultdict(set)
            for r_id in sorted(edge_types):
                t_range = range(0, self.args.nodes_all)
                valid_neg[r_id] = [[], []]
                valid_neg[r_id][0] = [-1] * len(self.valid_pos[r_id][0])
                valid_neg[r_id][1] = [-1] * len(self.valid_pos[r_id][1])
                idx = 0
                for h_id in self.valid_pos[r_id][0]:
                    neg_elem = -1
                    neg_paths = set()
                    valid_neg[r_id][0][idx] = h_id # valid_neg[r_id][0].append(h_id) # Corrupt only tails, path-aware corruption
                    t_id = self.valid_pos[r_id][1][idx]
                    neg_t = list(t_range) # random.randrange(t_range[0], t_range[1])
                    r_paths = self.val_ht2paths[tuple([h_id, tuple([r_id]), t_id])] if tuple([h_id, tuple([r_id]), t_id]) in self.val_ht2paths else None
                    pos_triplet = tuple([h_id, r_id, t_id])
                    pos_ent_set = set()
                    if r_paths is not None:
                        r_paths = r_paths.union(set([tuple([r_id])]))
                        for rel in r_paths:
                            pos_ent_set = pos_ent_set.union(self.subj2objs[h_id][rel])
                        neg_elem = None
                        max_iter = 10
                        while max_iter > 0:
                            neg_elem = random.choice(neg_t)
                            if (neg_elem not in pos_ent_set) : # and (neg_elem in self.obj2subjs and rel in self.obj2subjs[neg_elem] and len(self.obj2subjs[neg_elem][rel])>0 for rel in r_paths):
                                break
                            max_iter -= 1
                        if max_iter == 0:
                            neg_elem = random.choice(neg_t)
                    else:
                        neg_elem = random.choice(neg_t)
                    valid_neg[r_id][1][idx] = neg_elem # valid_neg[r_id][1].append(neg_elem) # no check for accidentally selecting the same tail entity??
                    neg_triplet = tuple([h_id, tuple([r_id]), neg_elem])
                    # neg_path = self.val_ht2paths(tuple([h_id, r_id, t_id]))
                    h_path = self.subj2objs[h_id].keys()
                    tn_path = self.obj2subjs[neg_elem].keys()
                    neg_paths = neg_paths.union(h_path)
                    neg_paths = neg_paths.union(tn_path)
                    if r_paths is not None:
                        neg_paths = neg_paths - r_paths
                        for r in r_paths:
                            elems_discard = []
                            for it in neg_paths:
                                if set(r).issubset(set(it)):
                                    elems_discard.append(it)
                            for it in elems_discard:
                                neg_paths.remove(it)
                    neg_paths = list(neg_paths)
                    random.shuffle(neg_paths)
                    neg_paths = set(neg_paths) if len(neg_paths) >= 1 else None
                    if neg_paths is not None:
                        valid_neg_paths[neg_triplet] = neg_paths
                    # print(pos_triplet, r_paths)
                    # print(neg_triplet, neg_paths)
                    idx += 1

            with open(file_name1, 'wb') as handle:
                pkl.dump(valid_neg, handle, protocol=pkl.HIGHEST_PROTOCOL)
            with open(file_name2, 'wb') as handle:
                pkl.dump(valid_neg_paths, handle, protocol=pkl.HIGHEST_PROTOCOL)
            return valid_neg, valid_neg_paths

    def load_links(self, args, path_name, file_name_link='link.dat', file_name_meta='meta.dat', type=None):
        file_name_link = type+".txt"
        edges_df = pd.read_csv(os.path.join(path_name, file_name_link), sep="\t", header=None)
        edges_df.columns = ["h_id", "r_id", "t_id"]

        ''' Sort the edges according to relation types '''
        df = edges_df.groupby('r_id', sort=True).apply(lambda x: x.reset_index(drop=True)).drop('r_id', axis=1).reset_index()
        entity_2id, rel_2id = dict(), dict()
        edges = {'total': 0, 'count': Counter(), 'meta': {}, 'attr': {},
                 'edges': defaultdict(dict), 'mat': defaultdict(list), 'adj': None}

        for index, row in df.iterrows():
            h_id, whole_r_id, t_id = str(row["h_id"]), str(row["r_id"]), str(row["t_id"])
            # indices = [pos for pos, char in enumerate(whole_r_id) if char == '/']
            r_id = whole_r_id #[indices[0]+1:indices[1]]
            if h_id not in entity_2id:
                entity_2id[h_id] = len(entity_2id.keys())
            if t_id not in entity_2id:
                entity_2id[t_id] = len(entity_2id.keys())
            if r_id not in rel_2id:
                rel_2id[r_id] = len(rel_2id.keys())

            link_weight = 1.0
            edges['attr'] = None
            if r_id not in edges['meta']:
                h_type, t_type = 1, 1
                edges['meta'][r_id] = dict()
                edges['meta'][r_id]["ht_type"] = (h_type, t_type)
                edges['meta'][r_id]["meta_features"] = None
            h_idx = entity_2id[h_id]
            t_idx = entity_2id[t_id]
            r_idx = rel_2id[r_id]
            edges['mat'][r_idx].append((h_idx, t_idx, link_weight))
            edges['edges'][r_idx][tuple([h_idx, t_idx])] = link_weight
            edges['count'][r_idx] += 1
            edges['total'] += 1

        self.entity_2id[type]["entity_2id"] = entity_2id
        self.rel_2id[type]["rel_2id"] = rel_2id

        new_data = {}
        for r_id in edges['mat']:
            new_data[r_id] = self.list_to_sp_mat(edges['mat'][r_id], type)
        edges['mat'] = new_data
        adjM = sum(edges['mat'].values())
        edges['adj'] = adjM

        return edges

    def load_links_tv(self, args, path_name, file_name_link='link.dat', file_name_meta='meta.dat', type=None):
        entity_2id, rel_2id = dict(), dict()
        link_weight = 1.0
        columns = ["h_id", "r_id", "t_id"]
        type = "tv"

        file_name_link = "train.txt"
        edges_df = pd.read_csv(os.path.join(path_name, file_name_link), sep="\t", header=None)
        edges_df.columns = columns
        df = edges_df.groupby('r_id', sort=True).apply(lambda x: x.reset_index(drop=True)).drop('r_id', axis=1).reset_index()
        edges_train = {'total': 0, 'count': Counter(), 'meta': {}, 'attr': {},
                 'edges': defaultdict(dict), 'mat': defaultdict(list), 'adj': None, 'mat_all': defaultdict(list), 'adj_all': None}
        for index, row in df.iterrows():
            h_id, r_id, t_id = str(row["h_id"]), str(row["r_id"]), str(row["t_id"])
            if h_id not in entity_2id:
                entity_2id[h_id] = len(entity_2id.keys())
            if t_id not in entity_2id:
                entity_2id[t_id] = len(entity_2id.keys())
            if r_id not in rel_2id:
                rel_2id[r_id] = len(rel_2id.keys())
            edges_train['attr'] = None
            if r_id not in edges_train['meta']:
                h_type, t_type = 1, 1
                edges_train['meta'][r_id] = dict()
                edges_train['meta'][r_id]["ht_type"] = (h_type, t_type)
                edges_train['meta'][r_id]["meta_features"] = None
            h_idx = entity_2id[h_id]
            t_idx = entity_2id[t_id]
            r_idx = rel_2id[r_id]
            edges_train['mat'][r_idx].append((h_idx, t_idx, link_weight))
            edges_train['edges'][r_idx][tuple([h_idx, t_idx])] = link_weight
            edges_train['count'][r_idx] += 1
            edges_train['total'] += 1
        self.entity_2id["train"]["entity_2id"] = entity_2id
        self.rel_2id["train"]["rel_2id"] = rel_2id

        file_name_link = "valid.txt"
        edges_df = pd.read_csv(os.path.join(path_name, file_name_link), sep="\t", header=None)
        edges_df.columns = columns
        df = edges_df.groupby('r_id', sort=True).apply(lambda x: x.reset_index(drop=True)).drop('r_id', axis=1).reset_index()
        edges_valid = {'total': 0, 'count': Counter(), 'meta': {}, 'attr': {},
                       'edges': defaultdict(dict), 'mat': defaultdict(list), 'adj': None}
        for index, row in df.iterrows():
            h_id, r_id, t_id = str(row["h_id"]), str(row["r_id"]), str(row["t_id"])
            if h_id not in entity_2id:
                entity_2id[h_id] = len(entity_2id.keys())
            if t_id not in entity_2id:
                entity_2id[t_id] = len(entity_2id.keys())
            if r_id not in rel_2id:
                rel_2id[r_id] = len(rel_2id.keys())
            edges_valid['attr'] = None
            if r_id not in edges_valid['meta']:
                h_type, t_type = 1, 1
                edges_valid['meta'][r_id] = dict()
                edges_valid['meta'][r_id]["ht_type"] = (h_type, t_type)
                edges_valid['meta'][r_id]["meta_features"] = None
            h_idx = entity_2id[h_id]
            t_idx = entity_2id[t_id]
            r_idx = rel_2id[r_id]
            edges_valid['mat'][r_idx].append((h_idx, t_idx, link_weight))
            edges_valid['edges'][r_idx][tuple([h_idx, t_idx])] = link_weight
            edges_valid['count'][r_idx] += 1
            edges_valid['total'] += 1

        self.entity_2id[type]["entity_2id"] = entity_2id
        self.rel_2id[type]["rel_2id"] = rel_2id

        tmp_train_edges = edges_train.copy()
        new_data = {}
        for r_id in edges_train['mat']:
            new_data[r_id] = self.list_to_sp_mat(edges_train['mat'][r_id], type) # not train
        edges_train['mat'] = new_data
        adjM = sum(edges_train['mat'].values())
        edges_train['adj'] = adjM

        # new_data = {}
        # for r_id in edges_valid['mat']:
        #     new_data[r_id] = self.list_to_sp_mat(edges_valid['mat'][r_id], type)
        # edges_valid['mat'] = new_data
        # adjM = sum(edges_valid['mat'].values())
        # edges_valid['adj'] = adjM

        file_name_link = "test.txt"
        edges_df = pd.read_csv(os.path.join(path_name, file_name_link), sep="\t", header=None)
        edges_df.columns = columns
        df = edges_df.groupby('r_id', sort=True).apply(lambda x: x.reset_index(drop=True)).drop('r_id', axis=1).reset_index()
        edges_test = {'total': 0, 'count': Counter(), 'meta': {}, 'attr': {},
                       'edges': defaultdict(dict), 'mat': defaultdict(list), 'adj': None}
        for index, row in df.iterrows():
            h_id, r_id, t_id = str(row["h_id"]), str(row["r_id"]), str(row["t_id"])
            if h_id not in entity_2id:
                entity_2id[h_id] = len(entity_2id.keys())
            if t_id not in entity_2id:
                entity_2id[t_id] = len(entity_2id.keys())
            if r_id not in rel_2id:
                rel_2id[r_id] = len(rel_2id.keys())
            edges_test['attr'] = None
            if r_id not in edges_test['meta']:
                h_type, t_type = 1, 1
                edges_test['meta'][r_id] = dict()
                edges_test['meta'][r_id]["ht_type"] = (h_type, t_type)
                edges_test['meta'][r_id]["meta_features"] = None
            h_idx = entity_2id[h_id]
            t_idx = entity_2id[t_id]
            r_idx = rel_2id[r_id]
            edges_test['mat'][r_idx].append((h_idx, t_idx, link_weight))
            edges_test['edges'][r_idx][tuple([h_idx, t_idx])] = link_weight
            edges_test['count'][r_idx] += 1
            edges_test['total'] += 1
        self.entity_2id["all"]["entity_2id"] = entity_2id
        self.rel_2id["all"]["rel_2id"] = rel_2id
        new_data = {}
        for r_id in edges_test['mat']:
            new_data[r_id] = self.list_to_sp_mat(edges_test['mat'][r_id], "all")
        edges_test['mat'] = new_data
        adjM = sum(edges_test['mat'].values())
        edges_test['adj'] = adjM

        new_data = {}
        for r_id in edges_valid['mat']:
            new_data[r_id] = self.list_to_sp_mat(edges_valid['mat'][r_id], "all")
        edges_valid['mat'] = new_data
        adjM = sum(edges_valid['mat'].values())
        edges_valid['adj'] = adjM

        new_data = {}
        for r_id in edges_train['mat']:
            new_data[r_id] = self.list_to_sp_mat(tmp_train_edges['mat'][r_id], "all") # not train
        edges_train['mat_all'] = new_data
        adjM = sum(edges_train['mat_all'].values())
        edges_train['adj_all'] = adjM

        return edges_train, edges_valid, edges_test

    def list_to_sp_mat(self, li, type="tv"):
        # (h, t, w)
        nodes = len(self.entity_2id[type]["entity_2id"].keys())
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(nodes, nodes)).tocsr()

