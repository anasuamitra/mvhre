import os
import argparse
import logging
import torch
import numpy as np
import random

from hin_data import hetero_data
from sampler_graphs import generate_subgraph_datasets
from hin_graph import SubgraphDataset
from utils_initialization import initialize_model
from utils_graph import collate_dgl, move_batch_to_device_dgl
from evaluator import Evaluator
from trainer import Trainer

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def link_predict(args):
    dl = hetero_data(args, os.path.join(args.output_dir, args.dataset))

    args.db_path = os.path.join(args.output_dir, args.dataset, 'contexts/')
    cond_1 = os.path.isdir(args.db_path)
    cond_2 = os.path.exists(os.path.join(args.db_path, 'data' + '.mdb'))
    t_file = os.path.join(args.output_dir, args.dataset, 'train_neg'+".pkl")
    v_file = os.path.join(args.output_dir, args.dataset, 'valid_neg'+".pkl")
    cond_3 = os.path.exists(t_file) and os.path.exists(v_file)
    if not (cond_1 and cond_2) and cond_3:
        generate_subgraph_datasets(args, dl)

    train_data = SubgraphDataset(args, args.db_path, dl, 'train_pos', 'train_neg', add_traspose_rels=args.add_traspose_rels,
                                 num_neg_samples_per_link=args.num_neg_samples_per_link)
    valid_data = SubgraphDataset(args, args.db_path, dl, 'valid_pos', 'valid_neg', add_traspose_rels=args.add_traspose_rels,
                                 num_neg_samples_per_link=args.num_neg_samples_per_link)
    args.num_rels = train_data.num_rels
    args.aug_num_rels = train_data.aug_num_rels
    args.inp_dim = train_data.n_feat_dim
    args.max_label_value = train_data.max_n_label
    args.num_bases = train_data.num_rels

    graph_classifier = initialize_model(args, dl, dgl_model, False) # load_model = False

    valid_evaluator = Evaluator(args, dl, graph_classifier, valid_data)

    trainer = Trainer(args, dl, graph_classifier, valid_data, train_data, valid_evaluator=valid_evaluator, train_evaluator=None)

    trainer.train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='PGRAIL testing for the LP datasets')
    parser.add_argument('--algorithm', type=str, default='PGRAIL')
    parser.add_argument('--dataset', type=str, default='fb4')
    parser.add_argument('--output_dir', type=str, default='../data/')

    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    parser.add_argument('--edge_hidden_dim', type=int, default=64)
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=64, help="Relation embedding size for attention")
    parser.add_argument('--num_layers', type=int, default=2) # both rgcn layers and transformer layers
    parser.add_argument('--num_clusters', type=int, default=25) # both rgcn layers and transformer layers
    parser.add_argument('--num_heads', type=int, default=2, help='Number of the attention heads. Default is 2.')
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cluster_dropout', type=float, default=0.75)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="Adam", help="Which optimizer to use?")
    parser.add_argument("--margin", default=10.0, type=float)
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument("--num_bases", "-b", type=int, default=-1,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--max_seq_length", default=12, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument('--test_ratio', default=0.1, type=float, help='Normalize or not')
    parser.add_argument("--max_links", type=int, default=1000000, help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--no_verbose", action="store_true")
    parser.add_argument("--num_neg_samples_per_link", type=int, default=1, help="Number of negative examples to sample per positive link")
    parser.add_argument('--add_traspose_rels', type=bool, default=False, help='whether to append adj matrix list with symmetric relations')

    parser.add_argument('--clustering', default=1, type=int, help='[-2, -1, 0, 1] Optimize via Clustering or not')
    parser.add_argument('--infomax', default=0, type=int, help='[0, 1] Optimize via InfoMax or not')

    # settings for relational context & paths
    parser.add_argument('--use_context', type=bool, default=True, help='whether to use relational context')
    parser.add_argument('--save_context', type=bool, default=False, help='whether to use relational context')
    parser.add_argument('--context_hops', type=int, default=4, help='"enclosing subgraph hop number"')
    parser.add_argument('--neighbor_samples', type=int, default=4, help='"enclosing subgraph hop number"')
    parser.add_argument('--use_path', type=bool, default=True, help='whether to use relational path')
    parser.add_argument('--walk_length', action='store', dest='walk_length', default=100, type=int, help=('length of each random walk'))
    parser.add_argument('--max_path_len', type=int, default=6, help='max length of the sliding window along a random-walk')
    parser.add_argument('--rw_repeat', type=int, default=10, help='max number of rw walks to perform from a source node')
    parser.add_argument('--path_samples', type=int, default=5, help='number of sampled paths between source & target nodes')

    parser.add_argument("--cluster_learning_coeff", default=1.0, type=float)
    parser.add_argument("--cluster_orthogonality_coeff", default=1.0, type=float)
    parser.add_argument("--cluster_size_coeff", default=1.0, type=float)
    parser.add_argument("--cluster_coeff", default=1.0, type=float)
    parser.add_argument("--summary_coeff", default=1.0, type=float)

    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--patience', type=int, default=40, help='Patience.') # num_train_epochs 300, patience 40
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", "-ne", type=int, default=300, help="Learning rate of the optimizer")
    parser.add_argument("--eval_every_iter", type=int, default=10, help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=10, help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument('--enable_cuda', default=True, type=bool, help='Enable CUDA')

    args = parser.parse_args()
    args.max_seq_length = args.max_path_len + 2
    args.neighbor_samples = args.max_seq_length - 3

    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:%d' % args.gpu)
    else:
        args.device = torch.device('cpu')

    if args.clustering == 1:
        from model.dgl.graph_classifier_1 import GraphClassifier as dgl_model
        args.infomax = 0
    elif args.clustering == -2:
        from model.dgl.graph_classifier_00 import GraphClassifier as dgl_model
        args.infomax = 0

    args.collate_fn = collate_dgl
    args.move_batch_to_device = move_batch_to_device_dgl

    args.save_str = args.dataset + "_h_" + str(args.hidden_dim) + "_lr_" + str(args.lr) + "_wd_" + str(args.weight_decay) + "_nh_" + str(args.num_heads) \
                    + "_nl_" + str(args.num_layers) + "_drp_" + str(args.dropout) + "_cdrp_" + str(args.cluster_dropout) \
                    + "_neigh_" + str(args.context_hops) + "_path_" + str(args.max_path_len) \
                    + "_l_" + str(args.cluster_learning_coeff) + "_o_" + str(args.cluster_orthogonality_coeff) + "_si_" + str(args.cluster_size_coeff) \
                    + "_s_" + str(args.summary_coeff) + "_c_" + str(args.cluster_coeff) + "_k_" + str(args.num_clusters) \
                    + "_cls_" + str(args.clustering) + "_infm_" + str(args.infomax)

    link_predict(args)

    '''
    1 / 54 python -W ignore main.py --dataset pubmed --clustering 1 --infomax 1 --hidden_dim 64 --lr 0.0005 --weight_decay 0.001 --num_heads 2 --num_layers 2 --dropout 0.1 --context_hops 4 --max_path_len 5 --path_samples 5 --cluster_learning_coeff 1.0 --cluster_orthogonality_coeff 1.0 --cluster_coeff 1.0 --summary_coeff 1.0 --num_clusters 25 --batch_size 128 --num_workers 4 --patience 40 --num_epochs 300 --gpu 0 
    '''