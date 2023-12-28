#!/usr/bin/env python
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C)2019 All rights reserved.
#
#   Author        : Eskimo
#   Email         : zhangfaninner@163.com
#   File Name     : utils.py
#   Last Modified : 2019-03-12 15:38
#   Describe      :
#
# ====================================================

import sys
# import os


import torch
import numpy as np
import time
import tensorflow as tf





def weight_init(m):

    if isinstance(m,torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data,0)

    if isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data,0)

    if isinstance(m,torch.nn.Embedding):
        torch.nn.init.orthogonal_(m.weight.data)

class TimeMeter:

    def __init__(self):
        self.start_time, self.duration, self.counter=0. ,0. ,0.

    def start(self):
        self.start_time=time.perf_counter()

    def stop(self):
        self.duration=time.perf_counter()-self.start_time
        self.counter+=1

    def get(self):
        return self.duration/self.counter

    def reset(self):
        self.start_time, self.duration, self.counter=0. ,0. ,0.
        







from io import BytesIO
import scipy.misc
class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        tag=self.tag(tag)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        tag=self.tag(tag)
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        tag=self.tag(tag)
        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
    def tag(self,tag):
        return tag.replace('.','/')
