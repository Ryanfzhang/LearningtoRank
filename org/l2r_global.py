#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Haitao Yu on 10/07/2018

"""Description

"""
import torch

class L2R_GLOBAL():
    ''' global attributes '''

    """ Seed """
    l2r_seed = 137

    """ GPU acceleration if expected """
    if torch.cuda.is_available():
        global_gpu, global_device = True, 'cuda'
    else:
        global_gpu, global_device = False, 'cpu'


