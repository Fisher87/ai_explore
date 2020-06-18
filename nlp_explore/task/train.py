#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：train.py
#   创 建 者：YuLianghua
#   创建日期：2020年04月20日
#   描    述：
#
#================================================================
import pdb

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tfdbg

from train_frame.bframe import TrainBaseFrame
from train_frame.data_process.data_processor import batch_iter
from train_frame.data_process.data_processor import DataProcessor
from tflags import TFlags

class Train(TrainBaseFrame):
    def __init__(self, model, flags, sess_config, field_len=2):
        super(Train, self).__init__(model, flags, sess_config)
        self.model = model
        self.flags = flags
        self.field_len = field_len
        self.sess_config = sess_config

    def get_batches(self, train_data, batch_size, num_epochs, shuffle=True):
        if len(train_data)==2:
            data = list(zip(train_data[0], train_data[1]))
        elif len(train_data)==3:
            data = list(zip(train_data[0], train_data[1], train_data[2]))
        batches = batch_iter(data, batch_size, num_epochs, shuffle=shuffle)
        return batches

    def get_feed_dict(self, batch_data, is_training=False, padding=True, samelen=False):
        '''
        @param: padding, whether to padding data in batch; If sequence data 
                length in batch is not same when do feed feed_dict will throw error;
        @param: samelen, whether `x`, 'y' padding to same length;
        '''
        label_batch = None
        if is_training:
            if self.field_len == 2:
                x_batch, y_batch = zip(*batch_data)
            elif self.field_len == 3:
                x_batch, y_batch, label_batch = zip(*batch_data)
        else:
            if self.field_len == 2:
                x_batch, y_batch = zip(*batch_data)
                # x_batch, y_batch = batch_data[0], batch_data[1]
            elif self.field_len == 3:
                x_batch, y_batch, label_batch = zip(*batch_data)
                # x_batch, y_batch, label_batch  = \
                #         batch_data[0], batch_data[1], batch_data[2]

        if padding:
            x_maxlen = max([len(x) for x in x_batch])
            y_maxlen = max([len(y) for y in y_batch])
            if samelen:
                x_maxlen = y_maxlen = max(x_maxlen, y_maxlen)
            _x_batch = [list(x)+[0]*(x_maxlen-len(x)) for x in x_batch]
            _y_batch = [list(y)+[0]*(y_maxlen-len(y)) for y in y_batch]
            x_batch = _x_batch
            y_batch = _y_batch

        if is_training:
            if label_batch is None:
                feed_dict = {
                        self.model.input_x : x_batch, 
                        self.model.input_y : y_batch,
                        self.model.dropout_keep_prob : self.flags.dropout_keep_prob
                            }
            else:
                feed_dict = {
                        self.model.input_x : x_batch, 
                        self.model.input_y : y_batch,
                        self.model.label   : label_batch,
                        self.model.dropout_keep_prob : self.flags.dropout_keep_prob
                            }

        else:
            if label_batch is None:
                feed_dict = {
                        self.model.input_x : x_batch, 
                        self.model.input_y : y_batch,
                        self.model.dropout_keep_prob : 1.0
                    }
            else:
                feed_dict = {
                        self.model.input_x : x_batch, 
                        self.model.input_y : y_batch,
                        self.model.label   : label_batch,
                        self.model.dropout_keep_prob : 1.0
                }

        return feed_dict
