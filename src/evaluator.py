import os
import numpy as np
import random
import pandas as pd
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict, Counter
from sklearn.metrics import f1_score, auc, roc_auc_score, precision_recall_curve, average_precision_score

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

dict_type = {"train": "t", "valid":"v", "test":"p", "test_2hop":"2", "test_random":"r"}

def calculate_scores(edge_list, confidence, labels):
    confidence = np.array(confidence)
    labels = np.array(labels)
    roc_auc = roc_auc_score(labels, confidence)
    auc_pr = average_precision_score(labels, confidence)
    mrr_list, mr_list, cur_mrr = [], [], 0
    hits_1, hits_3, hits_10 = [], [], []
    t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, h_id in enumerate(edge_list[0]):
        t_dict[h_id].append(edge_list[1][i])
        labels_dict[h_id].append(labels[i])
        conf_dict[h_id].append(confidence[i])
    for h_id in t_dict.keys():
        conf_array = np.array(conf_dict[h_id])
        rank = np.argsort(-conf_array)
        sorted_label_array = np.array(labels_dict[h_id])[rank]
        pos_index = np.where(sorted_label_array == 1)[0]
        if len(pos_index) == 0:
            continue
        pos_min_rank = np.min(pos_index)
        cur_mrr = 1 / (1 + pos_min_rank)
        mr_list.append(1 + pos_min_rank)
        mrr_list.append(cur_mrr)
        hits_1.append(1 + pos_min_rank <= 1)
        hits_3.append(1 + pos_min_rank <= 3)
        hits_10.append(1 + pos_min_rank <= 10)
    mr = np.mean(mr_list)
    mrr = np.mean(mrr_list)
    hits_1 = np.mean(hits_1)
    hits_3 = np.mean(hits_3)
    hits_10 = np.mean(hits_10)

    result_dict = OrderedDict()
    result_dict['roc_auc'] = roc_auc
    result_dict['auc_pr'] = auc_pr
    result_dict['mr'] = mr
    result_dict['mrr'] = mrr
    result_dict['hits@1'] = hits_1
    result_dict['hits@3'] = hits_3
    result_dict['hits@10'] = hits_10

    return result_dict

def writetofile_fn(args, res_2hop, res_random=None, data_type="test"):
    ''' Script to dump algorithm performance in results/ folder '''
    save_str = args.save_str
    columns = ["algo", "data", 'h', "eh", "ah", "lr", "wd", "nl", "nh", "ne", "p", "drp", "c_drp", "tr",
               "#ctx", "pl", "ps", "bs", "l", "o", "si", "s", "c", "k", "cls", "infm"]
    result_keys = ["roc_auc", "auc_pr", "mr", "mrr", "hits@1", "hits@3", "hits@10"]
    for i in result_keys:
        columns += [i]
    if res_random is not None:
        for i in result_keys:
            columns += [i]

    results_df = pd.DataFrame(columns=columns)
    temp = OrderedDict()
    temp["algo"] = args.algorithm
    temp["data"] = args.dataset
    temp["h"] = args.hidden_dim
    temp["eh"] = args.edge_hidden_dim
    temp["ah"] = args.attn_rel_emb_dim
    temp["lr"] = args.lr
    temp["wd"] = args.weight_decay
    temp["nl"] = args.num_layers
    temp["nh"] = args.num_heads
    temp["ne"] = args.num_epochs
    temp["p"] = args.patience
    temp["drp"] = args.dropout
    temp["c_drp"] = args.cluster_dropout
    temp["tr"] = args.test_ratio
    temp["#ctx"] = args.context_hops
    temp["pl"] = args.max_path_len
    temp["ps"] = args.path_samples
    temp["bs"] = args.batch_size
    temp["l"] = args.cluster_learning_coeff
    temp["o"] = args.cluster_orthogonality_coeff
    temp["si"] = args.cluster_size_coeff
    temp["s"] = args.summary_coeff
    temp["c"] = args.cluster_coeff
    temp["k"] = args.num_clusters
    temp["cls"] = args.clustering
    temp["infm"] = args.infomax

    for i in result_keys:
        temp[i] = res_2hop[i]
    if res_random is not None:
        for i in result_keys:
            temp[i] = res_random[i]
    results_df = results_df.append(temp, ignore_index=True)

    nsave_str = "results/" + args.algorithm + "_" + args.dataset + "_" + data_type
    if args.clustering == -4:
        nsave_str = nsave_str + "_000"
    elif args.clustering == -3:
        nsave_str = nsave_str + "_11"
    elif args.clustering == -2:
        nsave_str = nsave_str + "_00"
    elif args.clustering == -1:
        nsave_str = nsave_str + "_01"
    elif args.clustering == 0:
        nsave_str = nsave_str + "_0"
    elif args.clustering == 1:
        nsave_str = nsave_str + "_1"
    elif args.clustering == 2:
        nsave_str = nsave_str + "_10"
    elif args.clustering == 3:
        nsave_str = nsave_str + "_111"
    elif args.clustering == 4:
        nsave_str = nsave_str + "_4"
    with open(nsave_str + ".csv", 'a') as file:
        results_df.to_csv(file, index=False, header=file.tell() == 0)

class Evaluator():
    def __init__(self, params, dl, graph_classifier, data):
        self.params = params
        self.dl = dl
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, data_type='valid', save=False):
        assert data_type in ["train", "valid", "test", "test_2hop", "test_random"]
        pos_scores, pos_labels, neg_scores, neg_labels, s_logits, cluster_loss = [], [], [], [], [None, None], 0.0
        pos_head, pos_tail, neg_head, neg_tail, r_id = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                g, _, rel_labels = data_pos
                head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
                tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
                r = rel_labels.detach().cpu().tolist()
                h = g.ndata['idx'][head_ids].detach().cpu().tolist()
                t = g.ndata['idx'][tail_ids].detach().cpu().tolist()

                g1, _, rel_labels1 = data_neg
                head_ids1 = (g1.ndata['id'] == 1).nonzero().squeeze(1)
                tail_ids1 = (g1.ndata['id'] == 2).nonzero().squeeze(1)
                nr = rel_labels1.detach().cpu().tolist()
                nh = g1.ndata['idx'][head_ids1].detach().cpu().tolist()
                nt = g1.ndata['idx'][tail_ids1].detach().cpu().tolist()

                pos_head = np.concatenate([pos_head, np.array(h)])
                pos_tail = np.concatenate([pos_tail, np.array(t)])
                neg_head = np.concatenate([neg_head, np.array(nh)])
                neg_tail = np.concatenate([neg_tail, np.array(nt)])
                r_id = np.concatenate([r_id, np.array(r)])

                score_pos, score_neg, c_loss, s_logs, attn, mp, view_embeds, view_aggr, rep_rel = self.graph_classifier(data_pos, data_neg, self.dl, dict_type[data_type])

                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()
                if int(self.params.clustering) in [-3, 2, 3, 1, 4]:
                    cluster_loss += c_loss.detach().cpu()
                    if self.params.infomax == 1:
                        a, b = s_logs[0].shape
                        s_logits[0] = s_logs[0].view(a * b).detach().cpu().tolist()
                        s_logits[1] = s_logs[1].view(a * b).detach().cpu().tolist()

        # result_dict = metrics.calculate_scores(pos_scores + neg_scores, pos_labels + neg_labels)
        left = np.concatenate([pos_head, neg_head]).astype(int)
        right = np.concatenate([pos_tail, neg_tail]).astype(int)
        mid = np.concatenate([r_id, r_id]).astype(int)
        edge_list = np.concatenate([left.reshape((1, -1)), right.reshape((1, -1))], axis=0)
        result_dict = calculate_scores(edge_list, pos_scores + neg_scores, pos_labels + neg_labels)

        if save:
            writetofile_fn(self.params, result_dict, res_random=None, data_type=data_type)


        return (result_dict, pos_scores, neg_scores, cluster_loss, s_logits)
