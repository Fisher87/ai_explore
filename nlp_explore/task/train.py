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


# 1. classify task model;
from classify.textcnn import TextCNN 
from classify.bilstm_attention import BiLSTM_Attention

# 2. text match task model;
from text_match.semantic.DSSM import DSSM
# from text_match.semantic.ESIM import ESIM
# from text_match.semantic.BIMPM import BIMPM
# from text_match.semantic.ABCnn import ABCnn

# 3. chatbot/nlg task model;
from chatbot.Seq2SeqWithAtt import Seq2SeqWithAtt

argparser = argparse.ArgumentParser()
argparser.add_argument("-t", "--task", default="classify",
                       help="task name: [ classify | ner | text_match ]")
argparser.add_argument("-m", "--model", default="textcnn",
                       help="model type")
args = argparser.parse_args()

FLAGS = TFlags(args.task, args.model).FLAGS

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
                x_batch, y_batch = batch_data[0], batch_data[1]
            elif self.field_len == 3:
                x_batch, y_batch, label_batch  = \
                        batch_data[0], batch_data[1], batch_data[2]

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

# data processor
# chatbot data_processor
if args.task == 'classify':
    data_processor = DataProcessor(FLAGS.data_path, 
                                   maxlen=FLAGS.seq_len,
                                   vpath=FLAGS.vocab_path,
                                   slabel='\t')

if args.task == 'text_match':
    data_processor = DataProcessor(FLAGS.data_path, 
                                   ftype=3,
                                   maxlen=FLAGS.seq_len,
                                   vpath=FLAGS.vocab_path,
                                   slabel='\t')

if args.task == 'chatbot':
    data_processor = DataProcessor(FLAGS.data_path, 
                                   ftype=2,
                                   maxlen=25,
                                   padding=False,
                                   vpath=FLAGS.vocab_path,
                                   slabel='\t')
data_processor.load_data()

## split_data: {'train':['x', 'y'], 
#               'eval' :['x', 'y'], 
#               'test' :['x', 'y']}
splited_data = data_processor.data_split(eval=0.1, test=0.1)
train_data = splited_data['train']
eval_data  = splited_data['eval']
test_data  = splited_data['test']
field_len = len(train_data)

# init trainer
sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                       log_device_placement=False)
with tf.Graph().as_default():
    sess = tf.compat.v1.Session(config=sess_config)
    # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    with sess.as_default():
        # init model
        if args.task == 'classify':
            if args.model=='textcnn':
                model = TextCNN(FLAGS.seq_len,
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
                                        FLAGS.learning_rate)

        elif args.task == "text_match":
            if args.model == "dssm":
                model = DSSM(FLAGS.seq_len,
                             len(data_processor.char2idx),
                             FLAGS.num_classes,
                             FLAGS.embedding_dim,
                             FLAGS.random_embedding,
                             FLAGS.hidden_size,
                             FLAGS.learning_rate)
            elif args.model == "esim":
                model = ESIM()
            elif args.model == 'bimpm':
                model = BIMPM()
            elif args.model == 'abcnn':
                model = ABCnn()

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

        trainer = Train(model, FLAGS, sess_config, field_len=field_len)
        trainer.train(sess, train_data, eval_data, test_data)
