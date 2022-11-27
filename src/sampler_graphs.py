import os
import math
import struct
import logging
import random
import pickle as pkl
import pdb
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import sys
import torch
from scipy.special import softmax
from utils_dgl import _bfs_relational
from utils_graph import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def generate_subgraph_datasets(args, dl, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):
    testing = 'test_2hop' in splits or 'test_random' in splits or 'test' in splits

    if not testing:
        adj_list = dl.adj_list_all #dl.adj_list
        triplets = dict()
        triplets['train'], triplets['valid'] = list(), list()
        for idx, r_id in enumerate(sorted(dl.edges_train['edges'].keys())):
            for item in dl.edges_train['edges'][r_id]:
                triplets['train'].append((item[0], item[1], r_id))
        for idx, r_id in enumerate(sorted(dl.edges_val['edges'].keys())):
            for item in dl.edges_val['edges'][r_id]:
                triplets['valid'].append((item[0], item[1], r_id))
        triplets['train'] = np.array(triplets['train'])
        triplets['valid'] = np.array(triplets['valid'])

        valid_pos, valid_neg = [], []
        for r_id in sorted(dl.edges_val['edges'].keys()):
            if r_id in dl.edges_test['edges'].keys():
                idx = 0
                for h_id in dl.valid_pos[r_id][0]:
                    t_id = dl.valid_pos[r_id][1][idx]
                    nt_id = dl.valid_neg[r_id][1][idx]
                    valid_pos.append((h_id, t_id, r_id))
                    valid_neg.append((h_id, nt_id, r_id))
                    idx += 1
        valid_pos = np.array(valid_pos)
        valid_neg = np.array(valid_neg)

        train_pos, train_neg = [], []
        for r_id in sorted(dl.edges_train['edges'].keys()):
            if True: # r_id in dl.edges_test['edges'].keys():
                idx = 0
                for h_id in dl.train_pos[r_id][0]:
                    t_id = dl.train_pos[r_id][1][idx]
                    nt_id = dl.train_neg[r_id][1][idx]
                    train_pos.append((h_id, t_id, r_id))
                    train_neg.append((h_id, nt_id, r_id))
                    idx += 1
        train_pos = np.array(train_pos)
        train_neg = np.array(train_neg)

        graphs = {}
        graphs['valid'] = {'triplets': triplets['valid'], 'max_size': args.max_links, 'pos': valid_pos, 'neg': valid_neg}
        graphs['train'] = {'triplets': triplets['train'], 'max_size': args.max_links, 'pos': train_pos, 'neg': train_neg}

    else:
        adj_list = dl.adj_list_all
        triplets = dict()
        triplets['test'] = list()
        for idx, r_id in enumerate(sorted(dl.edges_test['edges'].keys())):
            for item in dl.edges_test['edges'][r_id]:
                triplets['test'].append((item[0], item[1], r_id))
        triplets['test'] = np.array(triplets['test'])

        if 'test' in splits:
            test_pos, test_neg = [], []
            for r_id in sorted(dl.edges_test['edges'].keys()):
                idx = 0
                for h_id in dl.test_pos[r_id][0]:
                    t_id = dl.test_pos[r_id][1][idx]
                    nt_id = dl.ntest_neg[r_id][1][idx]
                    test_pos.append((h_id, t_id, r_id))
                    test_neg.append((h_id, nt_id, r_id))
                    idx += 1
            test_pos = np.array(test_pos)
            test_neg = np.array(test_neg)
            # graphs = {}
            # graphs['test'] = {'triplets': triplets['test'], 'max_size': args.max_links, 'pos': test_pos, 'neg': test_neg}

        elif 'test_2hop' in splits:
            # triplets = dict()
            # triplets['test_2hop'] = list()
            # for idx, r_id in enumerate(sorted(dl.edges_test['edges'].keys())):
            #     for item in dl.edges_test['edges'][r_id]:
            #         triplets['test_2hop'].append((item[0], item[1], r_id))
            # triplets['test_2hop'] = np.array(triplets['test_2hop'])
            # print("triplets ", triplets)
            # print("dl.t_2_pos ", dl.t_2_pos)
            test_pos = dl.t_2_pos
            test_neg = dl.t_2_neg
            test_pos = np.array(test_pos)
            test_neg = np.array(test_neg)
            # graphs = {}
            # graphs['test_2hop'] = {'triplets': triplets['test_2hop'], 'max_size': args.max_links, 'pos': test_pos, 'neg': test_neg}

        elif 'test_random' in splits:
            # triplets = dict()
            # triplets['test_random'] = list()
            # for idx, r_id in enumerate(sorted(dl.edges_test['edges'].keys())):
            #     for item in dl.edges_test['edges'][r_id]:
            #         triplets['test_random'].append((item[0], item[1], r_id))
            # triplets['test_random'] = np.array(triplets['test_random'])
            test_pos = dl.t_r_pos
            test_neg = dl.t_r_neg
            test_pos = np.array(test_pos)
            test_neg = np.array(test_neg)
            # graphs = {}
            # graphs['test_random'] = {'triplets': triplets['test_random'], 'max_size': args.max_links, 'pos': test_pos, 'neg': test_neg}

        graphs = {}
        graphs['test'] = {'triplets': triplets['test'], 'max_size': args.max_links, 'pos': test_pos, 'neg': test_neg}

    # test_pos, test_neg = [], []
        # for r_id in sorted(dl.edges_test['edges'].keys()):
        #     idx = 0
        #     for h_id in dl.test_pos[r_id][0]:
        #         t_id = dl.test_pos[r_id][1][idx]
        #         if 'test_2hop' in splits:
        #             nt_id = dl.test_2hop_neg[r_id][1][idx]
        #         elif 'test_random' in splits:
        #             nt_id = dl.test_r_neg[r_id][1][idx]
        #         test_pos.append((h_id, t_id, r_id))
        #         test_neg.append((h_id, nt_id, r_id))
        #         idx += 1
        # print("graphs ", graphs)
    links2subgraphs(adj_list, graphs, args, max_label_value)

def links2subgraphs(A, graphs, args, max_label_value=None):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []
    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, args) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
    map_size = links_length * BYTES_PER_DATUM


    env = lmdb.open(args.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, args, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])
                enc_ratios.append(datum['enc_ratio'])
                num_pruned_nodes.append(datum['num_pruned_nodes'])

                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env)

        logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))

def get_average_subgraph_size(sample_size, links, A, params):
    # print("len(links) ", len(links))
    # print("sample_size ", sample_size)
    total_size = 0
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        nodes, n_labels, nodes_u, n_labels_u, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, params.context_hops, params.use_context, params.neighbor_samples)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels,
                 'nodes_u': nodes_u, 'n_labels_u': n_labels_u,
                 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size

def intialize_worker(A, params, max_label_value):
    global A_, params_, max_label_value_
    A_, params_, max_label_value_ = A, params, max_label_value

def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    nodes, n_labels, nodes_u, n_labels_u, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A_, params_.context_hops, params_.use_context, params_.neighbor_samples)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels,
             'nodes_u': nodes_u, 'n_labels_u': n_labels_u,
             'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)

def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)

def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)
    subgraph_nei_nodes_un = subgraph_nei_nodes_un - subgraph_nei_nodes_int

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    # list(root1_nei)  [160, 5249, 2437, 4746, 6412, 2381, 3375, 6832, 5522, 4980, 5367, 4921, 2553, 5181, 4894]
    # print("ind ", ind)
    # subgraph_h_nodes = [list(ind)[0]] + list(root1_nei)
    # subgraph_t_nodes = [list(ind)[1]] + list(root2_nei)
    # print("subgraph_h_nodes ", subgraph_h_nodes)
    # print("subgraph_t_nodes ", subgraph_t_nodes)
    subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    subgraph_u_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]
    subgraph_u = [adj[subgraph_u_nodes, :][:, subgraph_u_nodes] for adj in A_list]
    # subgraph_t = [adj[subgraph_t_nodes, :][:, subgraph_t_nodes] for adj in A_list]
    # print("subgraph_h ", subgraph_h)
    # print("subgraph_t ", subgraph_t)
    ''' ind  (3527, 5612)
        subgraph_h_nodes  [3527, 5252, 5126, 5319, 4905, 5612, 3532, 3565, 496, 6833, 404, 3546, 987, 3580, 3582]
        subgraph_t_nodes  [5612, 2592, 4097, 3617, 2177, 2885, 3527, 5319, 4905, 592, 4980, 6838, 4190]
        subgraph_h  [<15x15 sparse matrix of type '<class 'numpy.float64'>'
            with 3 stored elements in Compressed Sparse Row format>, <15x15 sparse matrix of type '<class 'numpy.float64'>'
            with 5 stored elements in Compressed Sparse Row format>, <15x15 sparse matrix of type '<class 'numpy.float64'>'
            with 6 stored elements in Compressed Sparse Row format>, <15x15 sparse matrix of type '<class 'numpy.float64'>'
            with 3 stored elements in Compressed Sparse Row format>, <15x15 sparse matrix of type '<class 'numpy.float64'>'
            with 5 stored elements in Compressed Sparse Row format>, <15x15 sparse matrix of type '<class 'numpy.float64'>'
            with 6 stored elements in Compressed Sparse Row format>]
        subgraph_t  [<13x13 sparse matrix of type '<class 'numpy.float64'>'
            with 1 stored elements in Compressed Sparse Row format>, <13x13 sparse matrix of type '<class 'numpy.float64'>'
            with 9 stored elements in Compressed Sparse Row format>, <13x13 sparse matrix of type '<class 'numpy.float64'>'
            with 2 stored elements in Compressed Sparse Row format>, <13x13 sparse matrix of type '<class 'numpy.float64'>'
            with 1 stored elements in Compressed Sparse Row format>, <13x13 sparse matrix of type '<class 'numpy.float64'>'
            with 10 stored elements in Compressed Sparse Row format>, <13x13 sparse matrix of type '<class 'numpy.float64'>'
            with 2 stored elements in Compressed Sparse Row format>]
    '''
    labels, enclosing_subgraph_nodes, labels_u, enclosing_subgraph_nodes_u = node_label(incidence_matrix(subgraph), incidence_matrix(subgraph_u), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels
    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    pruned_subgraph_nodes_u = np.array(subgraph_u_nodes)[enclosing_subgraph_nodes_u].tolist()
    pruned_labels_u = labels_u[enclosing_subgraph_nodes_u]
    if max_node_label_value is not None:
        pruned_labels_u = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels_u])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, pruned_subgraph_nodes_u, pruned_labels_u, \
           subgraph_size, enc_ratio, num_pruned_nodes


def node_label(subgraph, subgraph_u, max_distance=1):
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels
    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]


    roots_u = [0, 1]
    sgs_single_root_u = [remove_nodes(subgraph_u, [root]) for root in roots_u]
    dist_to_roots_u = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root_u)]
    dist_to_roots_u = np.array(list(zip(dist_to_roots_u[0][0], dist_to_roots_u[1][0])), dtype=int)
    target_node_labels_u = np.array([[0, 1], [1, 0]])
    labels_u = np.concatenate((target_node_labels_u, dist_to_roots_u)) if dist_to_roots_u.size else target_node_labels_u
    enclosing_subgraph_nodes_u = np.where(np.max(labels_u, axis=1) <= max_distance)[0]



    return labels, enclosing_subgraph_nodes, labels_u, enclosing_subgraph_nodes_u