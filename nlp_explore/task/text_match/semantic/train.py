#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：train.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月03日
#   描    述：
#
#================================================================
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import tensorflow as tf

from data import DataHandler
from DSSM import DSSM

data_handler = DataHandler('./data/vocab.txt', 
                           max_char_length=20)
# train data
train_path = './data/train.csv'
train_p, train_h, train_y = data_handler.load_data(train_path)

# dev data
dev_path = './data/dev.csv'
dev_p, dev_h, dev_y = data_handler.load_data(dev_path)

# test data
test_path = './data/test.csv'
test_p, test_h, test_y = data_handler.load_data(test_path)

# print(len(train_p), len(dev_p), len(test_p))
# print(train_p[0], dev_p[0], test_p[0])

sequence_length = data_handler.max_char_len
vocab_size = len(data_handler.vocab)
class_size = 2
embedding_dim=128
random_embedding=True
hidden_units = 128
learning_rate=0.001
batch_size = 1024
epochs = 50

model = DSSM(sequence_length, vocab_size)()

# with tf.Graph().as_default():
with tf.Session() as sess:
    # initialize all variables
    sess.run(tf.global_variables_initializer())

    def train(train_q, train_h, y):
        feed_dict = {
            model.query : train_q,
            model.doc : train_h, 
            model.y : y, 
            model.keep_prob: 0.5
        }
        loss, acc = sess.run([model.loss, model.acc], feed_dict)
        print(loss, acc)

    batches = data_handler.batch_iter(list(zip(train_p, train_h, train_y)), 
                                      batch_size, 
                                      epochs)
    for batch in batches:
        train_q_batch, train_h_batch, train_y_batch = zip(*batch)
        train(train_q_batch, train_h_batch, train_y_batch)
    
