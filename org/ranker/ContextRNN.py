#!/usr/bin/env python
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C)2019 All rights reserved.
#
#   Author        : Eskimo
#   Email         : zhangfaninner@163.com
#   File Name     : ContextRNN.py
#   Last Modified : 2019-09-02 22:31
#   Describe      :
#
# ====================================================



import os
import torch
import numpy as np
from torch import nn
from torch import optim
from org.l2r_global import L2R_GLOBAL
device = L2R_GLOBAL.global_device

class ContextRNN(object):
    def __init__(self, args, id= None, ranking_function = None, opt="Adam", lr = 1e-3, weight_decay =1e-3):
        self.ranking_function = rf(args)
        self.opt = opt
        self.lr = lr
        self.weight_decay = weight_decay
        self.init_optimizer()

    def reset_parameters(self):
        pass

    def init_optimizer(self):
        if 'Adam' == self.opt:
            self.optimizer = optim.Adam(self.ranking_function.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)  # use regularization
        elif 'RMS' == self.opt:
            self.optimizer = optim.RMSprop(self.ranking_function.get_parameters(), lr=self.lr, weight_decay=self.weight_decay)  # use regularization
        else:
            raise NotImplementedError
    def train(self, batch_ranks, batch_std , train=True, sp = False):
        self.ranking_function.train()
        ranking = self.ranking_function(batch_ranks, batch_std)

        if not sp:
            self.optimizer.zero_grad()
            self.ranking_function.loss.backward()
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            self.ranking_function.sp_loss.backward()
            self.optimizer.step()
        return self.ranking_function.loss

    def predict(self, batch_ranks, batch_std, train=False):
        self.ranking_function.eval()
        with torch.no_grad():
            ranking = self.ranking_function(batch_ranks, batch_std)
        return ranking 

    def save_model(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.ranking_function.state_dict(), dir+'/'+name+'.torch')

    def load(self, name):
        self.ranking_function= torch.load(name)
        

class rf(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gru = nn.GRUCell(args.hidden_size, args.hidden_size)
        self.fnn = nn.Sequential(
                nn.Linear(args.feature_size, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.Sigmoid(),
                )

        self.cnt_fnn = nn.Sequential(
                nn.Linear(2*args.feature_size, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.Sigmoid(),
                )

        self.com_fnn = nn.Sequential(
                nn.Linear(3* args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, 1)
                )

        self.h = torch.nn.Parameter(torch.zeros(args.hidden_size),requires_grad=False)


    def forward(self, batch_ranks, label):
        oral_feature_size = self.args.feature_size
        batch_doc_reprs = self.fnn(batch_ranks)

        bs, seq_len, feature_size = batch_doc_reprs.size()
        mask = torch.ones(bs, seq_len).bool().to(device)
        mask.requires_grad = False
        inf = torch.FloatTensor([float('-inf')]).unsqueeze(1).expand(bs, seq_len).to(device)
        inf.requires_grad = False

        pointer = torch.arange(seq_len).unsqueeze(0).expand(bs, -1).to(device)
        h = self.h.unsqueeze(0).expand(bs, -1)

        probs , reward, indices= [], [], []

        for i in range(seq_len):
            #doc 

            masked_batch_ranks = torch.masked_select(batch_ranks, mask.unsqueeze(2).expand(-1,-1,oral_feature_size)).view(bs, -1, oral_feature_size)
            batch_cnt = torch.cat([torch.max(masked_batch_ranks,dim =1)[0], torch.mean(masked_batch_ranks, dim=1)], dim=1)
            batch_cnt_reprs = self.cnt_fnn(batch_cnt)

            current_state = torch.cat([batch_doc_reprs, batch_cnt_reprs.unsqueeze(1).expand(-1, seq_len, -1), h.unsqueeze(1).expand(-1, seq_len, -1)], dim = 2)

            batch_outputs = self.com_fnn(current_state).squeeze(-1)
            batch_outputs = torch.nn.functional.softmax(batch_outputs, -1)
            batch_outputs = torch.where(mask , batch_outputs, inf)
            prob, indice = torch.max(batch_outputs, dim =-1)

            select_mask = (pointer==indice)
            probs.append(prob.unsqueeze(1))
            indices.append(indice.unsqueeze(1))
            select_label = torch.masked_select(label, select_mask).view(bs, -1)
            reward.append(self.get_reward(select_label, i))

            select_doc = torch.masked_select( batch_doc_reprs, select_mask.unsqueeze(2).expand(-1,-1, feature_size)).view(bs, feature_size)
            h = self.gru(select_doc, h)
            mask = mask * (~select_mask)

        assert torch.sum(mask)==0, print(mask)

        probs= torch.cat(probs, dim=1)
        rewards = torch.cat(reward, dim=1)

        rewards = rewards*(self.args.discount ** torch.arange(seq_len).unsqueeze(0).expand(bs, -1).float().to(device))
        R = torch.cumsum( rewards.flip((1)), dim =1).flip((1))
        self.loss = -torch.sum( R* torch.log(probs))

        sp_logits = torch.nn.functional.softmax(probs, dim =1)
        sp_stds = torch.nn.functional.softmax(label, dim=1)
        self.sp_loss = -torch.sum(sp_stds * torch.log(sp_logits))
        self.indices = torch.cat(indices, dim =1)

        return 0.9**(torch.sort(self.indices, dim = -1, descending=False)[1].float())


    def get_reward(self, label, i):
        return (2**label-1)/np.log2(i+2)

    def get_parameters(self):
        return list(self.parameters())
        


