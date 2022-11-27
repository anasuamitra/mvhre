import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import random
import pandas as pd
import time
import pickle
from collections import defaultdict, OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn import metrics
from tqdm import tqdm
from utils import EarlyStopping
from evaluator import Evaluator
from utils_initialization import initialize_model
from sampler_graphs import generate_subgraph_datasets
from hin_graph import SubgraphDataset

np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def calculate_scores(edge_list, confidence, labels):
    confidence = np.array(confidence)
    labels = np.array(labels)
    roc_auc = metrics.roc_auc_score(labels, confidence)
    auc_pr = metrics.average_precision_score(labels, confidence)
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

class Trainer():

    def choose_best(self, dict, data_type=''):
        result_keys = ["roc_auc", "auc_pr", "mr", "mrr", "hits@1", "hits@3", "hits@10"]
        if data_type == 'train':
            self.res_train["roc_auc"] = dict["roc_auc"] if dict["roc_auc"]>self.res_train["roc_auc"] else self.res_train["roc_auc"]
            self.res_train["auc_pr"] = dict["auc_pr"] if dict["auc_pr"]>self.res_train["auc_pr"] else self.res_train["auc_pr"]
            self.res_train["mr"] = dict["mr"] if dict["mr"]<self.res_train["mr"] else self.res_train["mr"]
            self.res_train["mrr"] = dict["mrr"] if dict["mrr"]>self.res_train["mrr"] else self.res_train["mrr"]
            self.res_train["hits@1"] = dict["hits@1"] if dict["hits@1"]>self.res_train["hits@1"] else self.res_train["hits@1"]
            self.res_train["hits@3"] = dict["hits@3"] if dict["hits@3"]>self.res_train["hits@3"] else self.res_train["hits@3"]
            self.res_train["hits@10"] = dict["hits@10"] if dict["hits@10"]>self.res_train["hits@10"] else self.res_train["hits@10"]
        elif data_type == 'valid':
            self.res_valid["roc_auc"] = dict["roc_auc"] if dict["roc_auc"]>self.res_valid["roc_auc"] else self.res_valid["roc_auc"]
            self.res_valid["auc_pr"] = dict["auc_pr"] if dict["auc_pr"]>self.res_valid["auc_pr"] else self.res_valid["auc_pr"]
            self.res_valid["mr"] = dict["mr"] if dict["mr"]<self.res_valid["mr"] else self.res_valid["mr"]
            self.res_valid["mrr"] = dict["mrr"] if dict["mrr"]>self.res_valid["mrr"] else self.res_valid["mrr"]
            self.res_valid["hits@1"] = dict["hits@1"] if dict["hits@1"]>self.res_valid["hits@1"] else self.res_valid["hits@1"]
            self.res_valid["hits@3"] = dict["hits@3"] if dict["hits@3"]>self.res_valid["hits@3"] else self.res_valid["hits@3"]
            self.res_valid["hits@10"] = dict["hits@10"] if dict["hits@10"]>self.res_valid["hits@10"] else self.res_valid["hits@10"]
        elif data_type == 'test_2hop':
            self.res_2hop["roc_auc"] = dict["roc_auc"] if dict["roc_auc"]>self.res_2hop["roc_auc"] else self.res_2hop["roc_auc"]
            self.res_2hop["auc_pr"] = dict["auc_pr"] if dict["auc_pr"]>self.res_2hop["auc_pr"] else self.res_2hop["auc_pr"]
            self.res_2hop["mr"] = dict["mr"] if dict["mr"]<self.res_2hop["mr"] else self.res_2hop["mr"]
            self.res_2hop["mrr"] = dict["mrr"] if dict["mrr"]>self.res_2hop["mrr"] else self.res_2hop["mrr"]
            self.res_2hop["hits@1"] = dict["hits@1"] if dict["hits@1"]>self.res_2hop["hits@1"] else self.res_2hop["hits@1"]
            self.res_2hop["hits@3"] = dict["hits@3"] if dict["hits@3"]>self.res_2hop["hits@3"] else self.res_2hop["hits@3"]
            self.res_2hop["hits@10"] = dict["hits@10"] if dict["hits@10"]>self.res_2hop["hits@10"] else self.res_2hop["hits@10"]
        elif data_type == 'test_random':
            self.res_random["roc_auc"] = dict["roc_auc"] if dict["roc_auc"]>self.res_random["roc_auc"] else self.res_random["roc_auc"]
            self.res_random["auc_pr"] = dict["auc_pr"] if dict["auc_pr"]>self.res_random["auc_pr"] else self.res_random["auc_pr"]
            self.res_random["mr"] = dict["mr"] if dict["mr"]<self.res_random["mr"] else self.res_random["mr"]
            self.res_random["mrr"] = dict["mrr"] if dict["mrr"]>self.res_random["mrr"] else self.res_random["mrr"]
            self.res_random["hits@1"] = dict["hits@1"] if dict["hits@1"]>self.res_random["hits@1"] else self.res_random["hits@1"]
            self.res_random["hits@3"] = dict["hits@3"] if dict["hits@3"]>self.res_random["hits@3"] else self.res_random["hits@3"]
            self.res_random["hits@10"] = dict["hits@10"] if dict["hits@10"]>self.res_random["hits@10"] else self.res_random["hits@10"]

    def __init__(self, params, dl, graph_classifier, valid_data, train_data, valid_evaluator=None, train_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.train_evaluator = Evaluator(params, dl, self.graph_classifier, train_data) #train_evaluator
        self.params = params
        self.dl = dl
        self.valid_data = valid_data
        self.train_data = train_data

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.weight_decay)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.weight_decay)

        self.ranking = nn.MarginRankingLoss(self.params.margin, reduction='mean')
        self.classifier = nn.BCEWithLogitsLoss()
        self.s_classifier = nn.BCEWithLogitsLoss()
        self.reset_training_state()

    def reset_training_state(self):
        self.early_stopping = EarlyStopping(patience=self.params.patience, verbose=True, save_path='checkpoints/pgrail_'+self.params.save_str+'.pt')
        self.res_train, self.res_valid, self.res_2hop, self.res_random = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
        self.res_train["roc_auc"], self.res_valid["roc_auc"], self.res_2hop["roc_auc"], self.res_random["roc_auc"] = -1e5, -1e5, -1e5, -1e5
        self.res_train["auc_pr"], self.res_valid["auc_pr"], self.res_2hop["auc_pr"], self.res_random["auc_pr"] = -1e5, -1e5, -1e5, -1e5
        self.res_train["mr"], self.res_valid["mr"], self.res_2hop["mr"], self.res_random["mr"] = 1e5, 1e5, 1e5, 1e5
        self.res_train["mrr"], self.res_valid["mrr"], self.res_2hop["mrr"], self.res_random["mrr"] = -1e5, -1e5, -1e5, -1e5
        self.res_train["hits@1"], self.res_valid["hits@1"], self.res_2hop["hits@1"], self.res_random["hits@1"] = -1e5, -1e5, -1e5, -1e5
        self.res_train["hits@3"], self.res_valid["hits@3"], self.res_2hop["hits@3"], self.res_random["hits@3"] = -1e5, -1e5, -1e5, -1e5
        self.res_train["hits@10"], self.res_valid["hits@10"], self.res_2hop["hits@10"], self.res_random["hits@10"] = -1e5, -1e5, -1e5, -1e5
        self.loss_train, self.loss_valid = defaultdict(), defaultdict()

    def train_epoch(self, epoch):
        total_loss, val_loss = 0, 0
        all_labels = []
        all_scores = []
        contexts = {}
        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        n_batches = int(len(self.train_data) / self.params.batch_size)
        # self.params.eval_every_iter = int(n_batches / 10)
        # self.params.eval_every_iter = int(self.params.batch_size/ 10)
        train_pos_head = np.array([])
        train_pos_tail = np.array([])
        train_neg_head = np.array([])
        train_neg_tail = np.array([])
        train_r_id = np.array([])

        self.graph_classifier.train()

        #try:
        for b_idx, batch in enumerate(dataloader):
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)

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

            train_pos_head = np.concatenate([train_pos_head, np.array(h)])
            train_pos_tail = np.concatenate([train_pos_tail, np.array(t)])
            train_neg_head = np.concatenate([train_neg_head, np.array(nh)])
            train_neg_tail = np.concatenate([train_neg_tail, np.array(nt)])
            train_r_id = np.concatenate([train_r_id, np.array(r)])

            score_pos, score_neg, cluster_loss, s_logits, attn, mp, view_embeds, view_aggr, rep_rel = self.graph_classifier(data_pos, data_neg, self.dl, "t")
            if self.params.save_context:
                if b_idx not in contexts:
                    contexts[b_idx]={}
                contexts[b_idx]['triplet']=list(zip(h, r, t))
                contexts[b_idx]['attn']=attn
                # for id in range(len(mp)):
                #     mp[id] = mp[id].cpu().detach().numpy()
                contexts[b_idx]['mp']=mp
                #contexts[b_idx]['view_embeds']=view_embeds
                #contexts[b_idx]['view_aggr']=view_aggr
                contexts[b_idx]['rep_rel']=rep_rel.cpu().detach().numpy()

            rk_loss = self.ranking(score_pos, score_neg.view(len(score_pos), -1).mean(dim=1), torch.Tensor([self.params.margin]).to(device=self.params.device))
            loss = rk_loss

            if int(self.params.clustering) in [-3, 2, 3, 1, 4]:
                loss += self.params.cluster_coeff * cluster_loss
                if int(self.params.infomax) == 1:
                    lbl_1 = torch.ones([s_logits[0].shape[0], ]) # Positive labels for samples
                    lbl_2 = torch.zeros([s_logits[0].shape[0], ]) # Negative labels for samples
                    lbl = torch.cat((lbl_1, lbl_2)) # lbl: bs x 2, s_logits: bs x 2
                    s_loss = self.s_classifier(
                        s_logits[0].view(-1).to(self.params.device),
                        lbl.view(-1).to(self.params.device)
                    )
                    loss += self.params.summary_coeff * s_loss
                    lbl_1 = torch.ones([s_logits[1].shape[0], ]) # Positive labels for samples
                    lbl_2 = torch.zeros([s_logits[1].shape[0], ]) # Negative labels for samples
                    lbl = torch.cat((lbl_1, lbl_2)) # lbl: bs x 2, s_logits: bs x 2
                    s_loss = self.s_classifier(
                        s_logits[1].view(-1).to(self.params.device),
                        lbl.view(-1).to(self.params.device)
                    )
                    loss += self.params.summary_coeff * s_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                all_scores += score_pos.squeeze().detach().cpu().tolist() + score_neg.squeeze().detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss


            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0 and ((b_idx > 0 and epoch == 1) or (epoch > 1)):
                with torch.no_grad():
                    tic = time.time()
                    result_dict, pos_scores, neg_scores, v_cluster_loss, vs_logits = self.valid_evaluator.eval(data_type="valid", save=False)
                    self.choose_best(result_dict, data_type='valid')

                    v_rk_loss = self.ranking(torch.Tensor(pos_scores).to(device=self.params.device), torch.Tensor(neg_scores).view(len(pos_scores), -1).mean(dim=1).to(device=self.params.device), torch.Tensor([self.params.margin]).to(device=self.params.device))

                    val_loss = v_rk_loss

                    if int(self.params.clustering) in [-3, 2, 3, 1, 4]:
                        val_loss += self.params.cluster_coeff * cluster_loss
                        if self.params.infomax == 1:
                            vlbl_1 = torch.ones([int(len(vs_logits[0])/2), ]) # Positive labels for samples
                            vlbl_2 = torch.zeros([int(len(vs_logits[0])/2), ]) # Negative labels for samples
                            vlbl = torch.cat((vlbl_1, vlbl_2)) # lbl: torch.Size([256]), s_logits: torch.Size([128, 2])
                            v_s_loss = self.s_classifier(
                                torch.Tensor(vs_logits[0]).to(self.params.device),
                                vlbl.view(-1).to(self.params.device)
                            )
                            val_loss += self.params.summary_coeff * v_s_loss
                            vlbl_1 = torch.ones([int(len(vs_logits[1])/2), ]) # Positive labels for samples
                            vlbl_2 = torch.zeros([int(len(vs_logits[1])/2), ]) # Negative labels for samples
                            vlbl = torch.cat((vlbl_1, vlbl_2)) # lbl: torch.Size([256]), s_logits: torch.Size([128, 2])
                            v_s_loss = self.s_classifier(
                                torch.Tensor(vs_logits[1]).to(self.params.device),
                                vlbl.view(-1).to(self.params.device)
                            )
                            val_loss += self.params.summary_coeff * v_s_loss

                    self.early_stopping(val_loss, self.graph_classifier, contexts, self.params.save_context)

                logging.info('\nValidation Performance: ' + ' val_loss = ' + str(val_loss) + ' roc_auc = ' + str(result_dict['roc_auc']) + ' auc_pr = ' + str(result_dict['auc_pr']) + ' mrr = ' + str(result_dict['mrr']) + ' hits@1 = ' + str(result_dict['hits@1']) + ' in ' + str(time.time() - tic))

                if self.early_stopping.early_stop:
                    print('Early stopping!')
                    break

        left = np.concatenate([train_pos_head, train_neg_head]).astype(int)
        right = np.concatenate([train_pos_tail, train_neg_tail]).astype(int)
        mid = np.concatenate([train_r_id, train_r_id]).astype(int)
        edge_list = np.concatenate([left.reshape((1, -1)), right.reshape((1, -1))], axis=0)
        result_dict = calculate_scores(edge_list, all_scores, all_labels)
        self.choose_best(result_dict, data_type='train')

        return total_loss

        #except:
        #    total_loss

    def train(self):

        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss = self.train_epoch(epoch)
            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with train loss: {loss} in {time_elapsed}')
            print()
            if self.early_stopping.early_stop:
                print('Early stopping!')
                break

        torch.cuda.empty_cache()
        # self.graph_classifier.load_state_dict(torch.load('checkpoints/pgrail_'+self.params.save_str+'.pt'))
        self.graph_classifier = torch.load('checkpoints/pgrail_'+self.params.save_str+'.pt')
        # print(self.graph_classifier.attention_weights)
        self.graph_classifier.eval()

        valid_evaluator = Evaluator(self.params, self.dl, self.graph_classifier, self.valid_data)
        train_evaluator = Evaluator(self.params, self.dl, self.graph_classifier, self.train_data)
        _, _, _, _, _ = valid_evaluator.eval(data_type="valid", save=True)
        _, _, _, _, _ = train_evaluator.eval(data_type="train", save=True)

        writetofile_fn(self.params, self.res_valid, res_random=None, data_type="best_valid")
        writetofile_fn(self.params, self.res_train, res_random=None, data_type="best_train")

        test(self.params, self.dl)

def test(params, dl):
    '''if args.dataset in ['dblp', 'imdb']:'''
    params.runs = 1 #10
    graph_classifier = initialize_model(params, dl, None, load_model=True)
    logging.info(f"Device: {params.device}")
    if len(dl.t_2_pos) > 0: # params.dataset not in ['dblp', 'imdb']:
        all_auc, all_auc_pr = [], []
        auc_mean, auc_pr_mean = 0, 0
        for r in range(1, params.runs + 1):
            params.db_path = os.path.join(params.output_dir, params.dataset, 'contexts', 'test_2hop')
            if not os.path.exists(os.path.join(params.db_path, 'data' + '.mdb')):
                generate_subgraph_datasets(params, dl, splits=['test_2hop'], saved_relation2id=None, max_label_value=graph_classifier.gnn.max_label_value)
            test_2hop = SubgraphDataset(params, params.db_path, dl, 'test_pos', 'test_neg', add_traspose_rels=params.add_traspose_rels,
                                        num_neg_samples_per_link=params.num_neg_samples_per_link)
            test_evaluator_2hop = Evaluator(params, dl, graph_classifier, test_2hop)
            result, _, _, _, _ = test_evaluator_2hop.eval(data_type="test_2hop", save=True)
            logging.info('\nTest Set Performance:' + str(result))

            all_auc.append(result['roc_auc'])
            auc_mean = auc_mean + (result['roc_auc'] - auc_mean) / r
            all_auc_pr.append(result['auc_pr'])
            auc_pr_mean = auc_pr_mean + (result['auc_pr'] - auc_pr_mean) / r
        logging.info('\nAvg test Set Performance -- mean auc :' + str(np.mean(all_auc)) + ' std auc: ' + str(np.std(all_auc)))
        logging.info('\nAvg test Set Performance -- mean auc_pr :' + str(np.mean(all_auc_pr)) + ' std auc_pr: ' + str(np.std(all_auc_pr)))

    if len(dl.t_r_pos) > 0:
        all_auc, all_auc_pr = [], []
        auc_mean, auc_pr_mean = 0, 0
        for r in range(1, params.runs + 1):
            params.db_path = os.path.join(params.output_dir, params.dataset, 'contexts', 'test_random')
            if not os.path.exists(os.path.join(params.db_path, 'data' + '.mdb')):
                generate_subgraph_datasets(params, dl, splits=['test_random'], saved_relation2id=None, max_label_value=graph_classifier.gnn.max_label_value)
            test_random = SubgraphDataset(params, params.db_path, dl, 'test_pos', 'test_neg', add_traspose_rels=params.add_traspose_rels,
                                          num_neg_samples_per_link=params.num_neg_samples_per_link)
            test_evaluator_random = Evaluator(params, dl, graph_classifier, test_random)

            result, _, _, _, _ = test_evaluator_random.eval(data_type="test_random", save=True)
            logging.info('\nTest Set Performance:' + str(result))

            all_auc.append(result['roc_auc'])
            auc_mean = auc_mean + (result['roc_auc'] - auc_mean) / r
            all_auc_pr.append(result['auc_pr'])
            auc_pr_mean = auc_pr_mean + (result['auc_pr'] - auc_pr_mean) / r
        logging.info('\nAvg test Set Performance -- mean auc :' + str(np.mean(all_auc)) + ' std auc: ' + str(np.std(all_auc)))
        logging.info('\nAvg test Set Performance -- mean auc_pr :' + str(np.mean(all_auc_pr)) + ' std auc_pr: ' + str(np.std(all_auc_pr)))

    if len(list(dl.ntest_neg.values())[0][0]) > 0:
        all_auc, all_auc_pr = [], []
        auc_mean, auc_pr_mean = 0, 0
        for r in range(1, params.runs + 1):
            params.db_path = os.path.join(params.output_dir, params.dataset, 'contexts', 'test')
            if not os.path.exists(os.path.join(params.db_path, 'data' + '.mdb')):
                generate_subgraph_datasets(params, dl, splits=['test'], saved_relation2id=None, max_label_value=graph_classifier.gnn.max_label_value)
            test_r = SubgraphDataset(params, params.db_path, dl, 'test_pos', 'test_neg', add_traspose_rels=params.add_traspose_rels,
                                          num_neg_samples_per_link=params.num_neg_samples_per_link)
            test_evaluator_r = Evaluator(params, dl, graph_classifier, test_r)

            result, _, _, _, _ = test_evaluator_r.eval(data_type="test", save=True)
            logging.info('\nTest Set Performance:' + str(result))

            all_auc.append(result['roc_auc'])
            auc_mean = auc_mean + (result['roc_auc'] - auc_mean) / r
            all_auc_pr.append(result['auc_pr'])
            auc_pr_mean = auc_pr_mean + (result['auc_pr'] - auc_pr_mean) / r
        logging.info('\nAvg test Set Performance -- mean auc :' + str(np.mean(all_auc)) + ' std auc: ' + str(np.std(all_auc)))
        logging.info('\nAvg test Set Performance -- mean auc_pr :' + str(np.mean(all_auc_pr)) + ' std auc_pr: ' + str(np.std(all_auc_pr)))


