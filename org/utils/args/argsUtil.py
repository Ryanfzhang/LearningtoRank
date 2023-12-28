#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 18/12/03 | https://y-research.github.io

"""Description

"""

import argparse

class ArgsUtil(object):
    def __init__(self, given_root=None):
        self.args_parser = argparse.ArgumentParser('Run ptl2r.')
        self.ini_l2r_args(given_root=given_root)

    def ini_l2r_args(self, given_root=None):
        self.given_root = given_root


        self.args_parser.add_argument('-framework', type=str, default='L2R', help='the specific learning-to-rank framework.')
        self.args_parser.add_argument('-epoch',type=int,default=300,help='the specific learning-to-rank framework.')
        self.args_parser.add_argument('-feature_size',type=int,default=46,help='the specific learning-to-rank framework.')
        self.args_parser.add_argument('-hidden_size',type=int,default=100,help='the specific learning-to-rank framework.')
        self.args_parser.add_argument('-dropout',type=float,default=0.1,help='the specific learning-to-rank framework.')
        self.args_parser.add_argument('-discount',type=float,default=0.93,help='the specific learning-to-rank framework.')
        self.args_parser.add_argument('-vali_per_n_step',type=int,default=1 ,help='the specific learning-to-rank framework.')

        ''' data '''
        self.args_parser.add_argument('-dataset', type=str, default="MQ2008_super",help='the data collection upon which you will perform learning-to-rank.')
        self.args_parser.add_argument('-dir_data',type=str, default="/home/ryan/data/MQ2008/",help='the path where the data locates.')

        ''' output '''
        self.args_parser.add_argument('-dir_output', type=str, default="/home/ryan/code/ranker/out/",help='the output path of the results.')

        ''' train '''
        self.args_parser.add_argument('-num_epoches', type=int, default=100, help='the number of training epoches.')
        self.args_parser.add_argument('-min_docs', type=int, default=10, help='the minimum size of a ranking.')
        self.args_parser.add_argument('-min_rele', type=int, default=1, help='the minimum number of relevant documents per ranking.')
        self.args_parser.add_argument('-unknown_as_zero', type=bool, default=True, help='semi completion')
        self.args_parser.add_argument('-fold', type=int, default=1, help='semi completion')






    def update_if_required(self, args):
        #if args.root_output != self.given_root:
        #    args.dir_output = args

        return args

    def get_l2r_args(self):
        l2r_args = self.args_parser.parse_args()
        l2r_args = self.update_if_required(l2r_args)
        return l2r_args

