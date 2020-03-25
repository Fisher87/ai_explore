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
import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import datetime
import numpy as np
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

    # define training procedure
    # grads_and_vars = model.optimier.compute_gradients(model.loss)

    # keep track of gradient values and sparsity
    # grad_summaries = []
    # for g, v in grads_and_vars:
    #     if g is not None:
    #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
    #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
    #         grad_summaries.append(grad_hist_summary)
    #         grad_summaries.append(sparsity_summary)
    # grad_summaries_merged = tf.summary.merge(grad_summaries)
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

    # summaries for loss and accuarcy
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary  = tf.summary.scalar("accuaracy", model.acc)
    # Train Summaries
    # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # 10 FLAGS.num_checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    # initialize all variables
    sess.run(tf.global_variables_initializer())

    def train(train_q, train_h, y):
        feed_dict = {
            model.query : train_q,
            model.doc : train_h, 
            model.y : y, 
            model.keep_prob: 0.5
        }
        _, loss, acc, step = sess.run([model.train_op, 
                                       model.loss, 
                                       model.acc, 
                                       model.global_step], 
                                       feed_dict)
        time_str = datetime.datetime.now().isoformat()
        avg_loss = np.sum(loss)/len(train_q)
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, avg_loss, acc))
        return loss

    def evaluate(eval_q, eval_h, eval_y):
        feed_dict = {
            model.query : eval_q,
            model.doc : eval_h, 
            model.y : eval_y, 
            model.keep_prob: 1.0
        }
        loss, acc, step = sess.run([model.loss, 
                                    model.acc, 
                                    model.global_step], 
                                    feed_dict)
        avg_loss = np.sum(loss)/len(eval_q)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, avg_loss, acc))
        return loss

    batches = data_handler.batch_iter(list(zip(train_p, train_h, train_y)), 
                                      batch_size, 
                                      epochs)
    train_losses = []
    eval_losses  = []
    best_loss    = 10000
    last_improvement = 0
    for batch in batches:
        train_q_batch, train_h_batch, train_y_batch = zip(*batch)
        train_loss = train(train_q_batch, train_h_batch, train_y_batch)
        current_step = tf.train.global_step(sess, model.global_step)

        # evaluation step
        if current_step % 10000==0:
            print("\n Evaluation:")
            eval_loss = evaluate(dev_p, dev_h, dev_y)
            if eval_loss < best_loss:
                best_loss = eval_loss
                last_improvement = 0
            else:
                last_improvement += 1

        # save checkpoint
        if current_step % 10000 == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("saved model checkpoint to {}\n".format(path))
    
        # early stopping
        if last_improvement >= 20:
            print("early stopping!!!")
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("saved model checkpoint to {}\n".format(path))
            break
