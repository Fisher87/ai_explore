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

from train_frame.bframe import TrainBaseFrame
from train_frame.data_process.data_processor import batch_iter
from train_frame.data_process.data_processor import DataProcessor
from tflags import TFlags


# classify task model
from classify.textcnn import TextCNN 
from classify.bilstm_attention import BiLSTM_Attention
# chatbot
from chatbot.Seq2SeqWithAtt import Seq2SeqWithAtt

argparser = argparse.ArgumentParser()
argparser.add_argument("-t", "--task", default="classify",
                       help="task name: [ classify | ner | text_match ]")
argparser.add_argument("-m", "--model", default="textcnn",
                       help="model type")
args = argparser.parse_args()

FLAGS = TFlags(args.task, args.model).FLAGS

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
                               ftype=2,
                               maxlen=25,
                               vpath=FLAGS.vocab_path,
                               slabel='\t')
data_processor.load_data()

## split_data: {'train':['x', 'y'], 
#               'eval' :['x', 'y'], 
#               'test' :['x', 'y']}
splited_data = data_processor.data_split(eval=0.1, test=0.1)
# print(splited_data)
train_data = splited_data['train']
eval_data  = splited_data['eval']
test_data  = splited_data['test']

# init trainer
sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                       log_device_placement=False)
with tf.Graph().as_default():
    sess = tf.compat.v1.Session(config=sess_config)
    with sess.as_default():
        # init model
        if args.task == 'classify':
            if args.model=='textcnn':
                model = TextCNN(FLAGS.pad_seq_len,
                                FLAGS.num_classes,
                                len(data_processor.char2idx),
                                FLAGS.embedding_dim,
                                FLAGS.learning_rate,
                                FLAGS.filter_sizes,
                                FLAGS.num_filters,
                                FLAGS.random_embedding,
                                FLAGS.l2_reg_lambda)
            elif args.model=='bilstm_attention':
                model = BiLSTM_Attention(FLAGS.seq_len,
                                        len(data_processor.char2idx),
                                        FLAGS.embedding_dim,
                                        FLAGS.hidden_size,
                                        FLAGS.attention_size,
                                        FLAGS.num_classes,
                                        FLAGS.learning_rate,
                                        )
        elif args.task == "text_match":
            if args.model == "":
                pass

        elif args.task == "chatbot":
            if args.model == "seq2seq_att":
                word2inx = data_processor.char2idx
                word2inx['<GO>'] = len(word2inx) + 1
                word2inx['<EOS>']= len(word2inx) + 1
                model = Seq2SeqWithAtt(FLAGS.max_len,
                                       len(word2inx),  # FLAGS.vocab_size,
                                       word2inx,       # FLAGS.word2inx,
                                       FLAGS.embedding_dim,
                                       FLAGS.state_size,
                                       FLAGS.num_layers,
                                       FLAGS.use_attention,
                                       FLAGS.use_teacher_forcing,
                                       FLAGS.learning_rate,
                                       FLAGS.beam_width
                                       )

        trainer = Train(model, FLAGS, sess_config)
        trainer.train(sess, train_data, eval_data, test_data)
