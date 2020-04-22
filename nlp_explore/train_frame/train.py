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
import numpy as np
import tensorflow as tf

from flags import FLAGS
from textcnn import TextCNN 
from bframe import TrainBaseFrame
from data_process.data_processor import batch_iter
from data_process.data_processor import DataProcessor


class Train(TrainBaseFrame):
    def __init__(self, model, flags, sess_config):
        super(Train, self).__init__(model, flags, sess_config)
        self.model = model
        self.flags = flags
        self.sess_config = sess_config

    def get_batches(self, train_data, batch_size, num_epochs, shuffle=True):
        if len(train_data)==2:
            data = list(zip(train_data[0], train_data[1]))
        elif len(train_data)==3:
            data = list(zip(train_data[0], train_data[1], train_data[2]))
        batches = batch_iter(data, batch_size, num_epochs, shuffle=shuffle)
        return batches

    def get_feed_dict(self, batch_data, is_training=False):
        if is_training:
            x_batch, y_batch = zip(*batch_data)
            feed_dict = {
                    self.model.input_x : x_batch, 
                    self.model.input_y : y_batch,
                    self.model.dropout_keep_prob : self.flags.dropout_keep_prob
                        }
        else:
            x_batch, y_batch = batch_data[0], batch_data[1]
            feed_dict = {
                    self.model.input_x : x_batch, 
                    self.model.input_y : y_batch,
                    self.model.dropout_keep_prob : 1.0
                }

        return feed_dict

# data processor
data_processor = DataProcessor(FLAGS.data_path, 
                               vpath=FLAGS.vocab_path,
                               slabel='\t')
data_processor.load_data()
## split_data: {'train':['x', 'y'], 'eval':['x', 'y'], 'test':['x', 'y']}
splited_data = data_processor.data_split(eval=0.1, test=0.1)
train_data = splited_data['train']
eval_data   = splited_data['eval']
test_data  = splited_data['test']

# init trainer
sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                       log_device_placement=False)
with tf.Graph().as_default():
    sess = tf.compat.v1.Session(config=sess_config)
    with sess.as_default():
        # init model
        model = TextCNN(FLAGS.pad_seq_len,
                        FLAGS.num_classes,
                        len(data_processor.char2idx),
                        FLAGS.embedding_dim,
                        FLAGS.learning_rate,
                        FLAGS.filter_sizes,
                        FLAGS.num_filters,
                        FLAGS.random_embedding,
                        FLAGS.l2_reg_lambda)
        trainer = Train(model, FLAGS, sess_config)
        trainer.train(sess, train_data, eval_data, test_data)
