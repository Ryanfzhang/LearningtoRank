#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C)2019 All rights reserved.
#
#   Author        : Eskimo
#   Email         : zhangfaninner@163.com
#   File Name     : l2r.py
#   Last Modified : 2019-09-03 17:06
#   Describe      :
#
# ====================================================

import sys
# import os

import os
import sys
import datetime
import numpy as np
import torch
from tqdm import tqdm 

from org.utils.bigdata.BigPickle import pickle_save, pickle_load
from org.data import data_ms
from org.data.data_utils import get_data_loader
from org.eval.metric import tor_nDCG_at_ks, tor_ap_at_ks
from org.ranker.ContextRNN import ContextRNN
from org.ranker.rnn import MDP
from org.ranker.attBasedRnn import attRNN
from org.l2r_global import L2R_GLOBAL
from org.utils.utils import TimeMeter
from org.utils.args.argsUtil import ArgsUtil

device = L2R_GLOBAL.global_device


def train(args):
    if torch.cuda.is_available(): print('--GPU is launched--') 

    dataset, dir_data= args.dataset, args.dir_data
    fold_num = args.fold
    query_level_scale, unknown_as_zero, min_docs, min_rele = False, True, args.min_docs, args.min_rele
    num_features, has_comment, multi_level_rele, max_rele_level = data_ms.get_data_meta(dataset= dataset)
    args.feature_size = num_features
    ranker = attRNN(args, "attRNN")

    dir_fold_k = dir_data+"Fold"+str(fold_num)+"/"
    file_train, file_vali, file_test = dir_fold_k + 'train.txt', dir_fold_k + 'vali.txt', dir_fold_k + 'test.txt'
    
    train_data_loader = get_data_loader(original_file=file_train, has_comment=has_comment, query_level_scale=query_level_scale,
                                        min_docs=min_docs, min_rele=min_rele, need_pre_sampling=True, sample_times_per_q=1, shuffle = True, batch_size = 1,
                                        unknown_as_zero=unknown_as_zero, binary_rele=False)

    test_data_loader = get_data_loader(original_file=file_test, has_comment=has_comment, query_level_scale=query_level_scale,
                                       min_docs=min_docs, min_rele=min_rele, need_pre_sampling=False, sample_times_per_q=1, shuffle = False, batch_size = 1, binary_rele=False) 
    vali_data_loader = get_data_loader(original_file=file_vali, has_comment=has_comment, query_level_scale=query_level_scale,
                                       min_docs=min_docs, min_rele=min_rele, need_pre_sampling=False, sample_times_per_q=1, shuffle = False, batch_size = 1, binary_rele=False)

    
    # to device
    ranker.ranking_function.to(device)
    print('Fold-', fold_num)
    best_test_ndcg = float('-inf') 
    log_file=open("./out/{}_{}.txt".format(args.dataset,args.fold),"a+")


    for epoch in tqdm(range(args.epoch)):
        datas = tqdm(train_data_loader, total= len(train_data_loader))
        for entry in datas:
            tor_batch_rankings , tor_batch_stds = entry[0].squeeze(0).to(device), entry[1].squeeze(0).to(device)
            
            '''
            if epoch<10:
                batch_loss = ranker.train(batch_ranks=tor_batch_rankings, batch_std= tor_batch_stds, sp=True)
            else:
                batch_loss = ranker.train(batch_ranks=tor_batch_rankings, batch_std= tor_batch_stds, sp=False)
            '''
            batch_loss = ranker.train(batch_ranks=tor_batch_rankings, batch_std= tor_batch_stds, sp=False)
            datas.set_description("loss: {:.4f}".format(batch_loss))

        if epoch % args.vali_per_n_step ==0:
            #test
            test_ndcg_ks = []
            for entry in test_data_loader:
                tor_batch_rankings , tor_batch_stds = entry[0].to(device), entry[1].squeeze().to(device)
                tor_rele_pred = ranker.predict(batch_ranks=tor_batch_rankings, batch_std=tor_batch_stds.unsqueeze(0))
                tor_rele_pred = tor_rele_pred.squeeze().data.cpu()
                _, tor_sorted_inds = torch.sort(tor_rele_pred, descending=True)

                sys_sorted_labels = tor_batch_stds[tor_sorted_inds.cpu()]
                ideal_sorted_labels, _ = torch.sort(tor_batch_stds.squeeze(), descending=True)

                ndcg_at_k = tor_nDCG_at_ks(sys_sorted_labels.cpu(), ideal_sorted_labels.cpu(), [1,3,5,7,10], True)
                test_ndcg_ks.append(ndcg_at_k)

            test_ndcg = torch.stack(test_ndcg_ks, dim =0)
            test_ndcg = torch.mean(test_ndcg, dim =0)
            tqdm.write("test ndcg 1:{:.4f}, 3:{:.4f}, 5:{:.4f}, 7:{:.4f}, 10:{:.4f}".format(*test_ndcg.tolist()))
            print("test ndcg 1:{:.4f}, 3:{:.4f}, 5:{:.4f}, 7:{:.4f}, 10:{:.4f}".format(*test_ndcg.tolist()), file = log_file)
            if(test_ndcg.tolist()[-1]>best_test_ndcg):
                best_test_ndcg = test_ndcg.tolist()[-1]

            vali_ndcg_ks = []
            for entry in vali_data_loader:
                tor_batch_rankings , tor_batch_stds = entry[0].to(device), entry[1].squeeze().to(device)
                tor_rele_pred = ranker.predict(batch_ranks=tor_batch_rankings, batch_std=tor_batch_stds.unsqueeze(0))
                tor_rele_pred = tor_rele_pred.squeeze()
                _, tor_sorted_inds = torch.sort(tor_rele_pred, descending=True)
                sys_sorted_labels = tor_batch_stds[tor_sorted_inds]
                ideal_sorted_labels, _ = torch.sort(tor_batch_stds.squeeze(), descending=True)

                ndcg_at_k = tor_nDCG_at_ks(sys_sorted_labels.cpu(), ideal_sorted_labels.cpu(), [1,3,5,7,10], True)
                vali_ndcg_ks.append(ndcg_at_k)

            vali_ndcg = torch.stack(vali_ndcg_ks, dim =0)
            vali_ndcg = torch.mean(vali_ndcg, dim =0)
            tqdm.write("vali ndcg 1:{:.4f}, 3:{:.4f}, 5:{:.4f}, 7:{:.4f}, 10:{:.4f}".format(*vali_ndcg.tolist()))
            

    print("finish! Fold {} best test ndcg10:{:.4f}".format(fold_num, best_test_ndcg))
    print("finish! Fold {} best test ndcg10:{:.4f}".format(fold_num, best_test_ndcg), file = log_file)

if __name__=="__main__":
    argsUtil = ArgsUtil()
    args = argsUtil.get_l2r_args()

    train(args)
