import os
import pickle as pkl
from collections import defaultdict, Counter
from operator import itemgetter
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import itertools
import networkx as nx
import torch
import dgl
import matplotlib.pyplot as plt

import utils_network

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def get_path_dict_and_length(rel2paths, null_relation, max_path_len):
    path2id = {}
    id2path = []
    id2length = []
    n_paths = 0
    for path in rel2paths:
        if path not in path2id:
            path2id[path] = n_paths
            id2length.append(len(path))
            id2path.append(tuple(list(path) + [null_relation] * (max_path_len - len(path))))  # padding
            n_paths += 1
    return path2id, id2path, id2length

def get_sparse_feature_matrix(non_zeros, n_cols):
    features = sp.lil_matrix((len(non_zeros), n_cols), dtype=np.float64)
    for i in range(len(non_zeros)):
        for j in non_zeros[i]:
            features[i, j] = +1.0
    return features

def one_hot_path_id(train_paths, path_dict): #path2id
    res = []
    for (head, relation, tail) in train_paths:
        bop_list = []
        for paths in train_paths[tuple([head, relation, tail])]:
            bop_list.append(path_dict[paths])
        res.append(bop_list)
    return get_sparse_feature_matrix(res, len(path_dict))

def sample_paths(dl, args):
    print('Sampling the knowledge graph ...')
    null_relation = len(list(dl.rel_2id["all"]["rel_2id"].values()))
    edges = []
    for r_id in dl.edges_train['edges']:
        for k, v in dl.edges_train['edges'][r_id].items():
            edges.append((k[0], k[1], r_id, v))

    directory = os.path.join(dl.args.output_dir, dl.args.dataset, 'paths/')
    length = str(dl.args.max_path_len)

    if os.path.exists(os.path.join(directory, 'ht2paths_' + length + '.pkl')):
        ''' Checking for pre-existing sampled random-walks. '''
        print('Loading paths from files ...')
        rel2paths = pkl.load(open(os.path.join(directory, 'rel2paths_' + length + '.pkl'), 'rb'))
        ht2paths = pkl.load(open(os.path.join(directory, 'ht2paths_' + length + '.pkl'), 'rb'))
        r_ht2paths = pkl.load(open(os.path.join(directory, 'r_ht2paths_' + length + '.pkl'), 'rb'))
    else:
        g = utils_network.HIN()
        for idx, item in enumerate(edges):
            g.add_edge(int(item[0]), 1, int(item[1]), 1, int(item[2]), weight=item[3])
            if dl.args.dataset in ['ddb', 'lastfm']:
                g.add_edge(int(item[1]), 1, int(item[0]), 1, int(item[2]), weight=item[3])
        for idx, item in enumerate(edges):
            g.add_edge(int(item[0]), 1, int(item[1]), 1, int(item[2]), weight=item[3])
            if dl.args.dataset in ['ddb', 'lastfm']:
                g.add_edge(int(item[1]), 1, int(item[0]), 1, int(item[2]), weight=item[3])
        for idx, item in enumerate(edges):
            g.add_edge(int(item[0]), 1, int(item[1]), 1, int(item[2]), weight=item[3])
            if dl.args.dataset in ['ddb', 'lastfm']:
                g.add_edge(int(item[1]), 1, int(item[0]), 1, int(item[2]), weight=item[3])

        g.print_statistics()

        tmp_walk_fname = os.path.join(directory, 'random_walk_' + length + '.txt')
        with open(tmp_walk_fname, 'w') as f:
            for walk in g.random_walks(dl.args.rw_repeat, dl.args.walk_length):
                f.write('%s\n' % ' '.join(map(str, walk)))

        rel2paths = defaultdict(set)
        ht2paths, r_ht2paths = defaultdict(set), defaultdict(set)
        num_lines = sum(1 for line in open(tmp_walk_fname, 'r'))
        with open(tmp_walk_fname, 'r') as f:
            for line in tqdm(f, total=num_lines):
                tokens = line.strip().split()
                for w in range(dl.args.max_path_len):
                    for i in range(0, len(tokens)-w*2, 2):
                        li = tokens[i:i+1+w*2]
                        li = [int(item) for item in li]
                        if len(li) > 1:
                            # count += 1
                            head = li[0]
                            tail = li[-1]
                            rel_li = li[1:-1]
                            rel_li = [rel_li[o] for o in range(0, len(rel_li), 2)]
                            if len(rel_li)==1: # Checking for single relations aka training triplets
                                r_ht2paths[tuple([head, tail])].add(tuple([i for i in rel_li]))
                            rel2paths[tuple([i for i in rel_li])].add(tuple([head, tail]))
                            ht2paths[tuple([head, tail])].add(tuple([i for i in rel_li]))
        pkl.dump(rel2paths, open(os.path.join(directory, 'rel2paths_' + length + '.pkl'), 'wb'))
        pkl.dump(ht2paths, open(os.path.join(directory, 'ht2paths_' + length + '.pkl'), 'wb'))
        pkl.dump(r_ht2paths, open(os.path.join(directory, 'r_ht2paths_' + length + '.pkl'), 'wb'))

    if os.path.exists(os.path.join(directory, 'train_ht2paths' + '.pkl')):
        ''' Checking for pre-existing sampled random-walks. '''
        print('Loading train/ validation/ test paths from files ...')
        train_ht2paths = pkl.load(open(os.path.join(directory, 'train_ht2paths' + '.pkl'), 'rb'))
        valid_ht2paths = pkl.load(open(os.path.join(directory, 'valid_ht2paths' + '.pkl'), 'rb'))
        test_ht2paths = pkl.load(open(os.path.join(directory, 'test_ht2paths' + '.pkl'), 'rb'))
    else:
        train_ht2paths, valid_ht2paths, test_ht2paths = defaultdict(set), defaultdict(set), defaultdict(set)
        for r_id in dl.edges_train['edges'].keys():
            for (h_id, t_id) in dl.edges_train['edges'][r_id]:
                if (h_id, t_id) in ht2paths.keys():
                    s_set = ht2paths[tuple([h_id, t_id])] - set([tuple([r_id])])
                    if (len(s_set)) >= 1:
                        train_ht2paths[tuple([h_id, tuple([r_id]), t_id])] = s_set
        for r_id in dl.edges_val['edges'].keys():
            for (h_id, t_id) in dl.edges_val['edges'][r_id]:
                if (h_id, t_id) in ht2paths.keys():
                    s_set = ht2paths[tuple([h_id, t_id])] - set([tuple([r_id])])
                    if (len(s_set)) >= 1:
                        valid_ht2paths[tuple([h_id, tuple([r_id]), t_id])] = s_set
        for r_id in dl.edges_test['edges'].keys():
            for (h_id, t_id) in dl.edges_test['edges'][r_id]:
                if (h_id, t_id) in ht2paths.keys():
                    s_set = ht2paths[tuple([h_id, t_id])] - set([tuple([r_id])])
                    if (len(s_set)) >= 1:
                        test_ht2paths[tuple([h_id, tuple([r_id]), t_id])] = s_set
        pkl.dump(train_ht2paths, open(os.path.join(directory, 'train_ht2paths' + '.pkl'), 'wb'))
        pkl.dump(valid_ht2paths, open(os.path.join(directory, 'valid_ht2paths' + '.pkl'), 'wb'))
        pkl.dump(test_ht2paths, open(os.path.join(directory, 'test_ht2paths' + '.pkl'), 'wb'))

    n_original_edges, n_original_edges_valid, n_original_edges_test = 0, 0, 0
    n_path_edges, n_path_edges_valid, n_path_edges_test = 0, 0, 0
    for r_id in dl.edges_train['edges'].keys():
        for (h_id, t_id) in dl.edges_train['edges'][r_id]:
            n_original_edges += 1
    for r_id in dl.edges_val['edges'].keys():
        for (h_id, t_id) in dl.edges_val['edges'][r_id]:
            n_original_edges_valid += 1
    for r_id in dl.edges_test['edges'].keys():
        for (h_id, t_id) in dl.edges_test['edges'][r_id]:
            n_original_edges_test += 1
    for _ in train_ht2paths.keys():
        n_path_edges += 1
    for _ in valid_ht2paths.keys():
        n_path_edges_valid += 1
    for _ in test_ht2paths.keys():
        n_path_edges_test += 1
    print(n_original_edges, n_original_edges_valid, n_original_edges_test)
    print(n_path_edges, n_path_edges_valid, n_path_edges_test)

    dl.path2id, dl.id2path, dl.id2length = get_path_dict_and_length(rel2paths, null_relation, dl.args.max_path_len)
    print('transforming paths to one hot IDs ...')
    dl.train_paths = one_hot_path_id(train_ht2paths, dl.path2id)
    dl.valid_paths = one_hot_path_id(valid_ht2paths, dl.path2id)
    dl.test_paths = one_hot_path_id(test_ht2paths, dl.path2id)
    dl.train_ht2paths = train_ht2paths
    dl.valid_ht2paths = valid_ht2paths
    dl.test_ht2paths = test_ht2paths
    dl.num_mp =  len(list(dl.path2id.keys()))

