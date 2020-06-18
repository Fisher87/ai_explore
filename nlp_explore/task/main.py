#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：main.py
#   创 建 者：YuLianghua
#   创建日期：2020年06月18日
#   描    述：
#
#================================================================
import os
import pdb

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tfdbg

from train_frame.bframe import TrainBaseFrame
from train_frame.data_process.data_processor import batch_iter
from train_frame.data_process.data_processor import DataProcessor
from tflags import TFlags

# text match task model;
# 1. classify task model;
from classify.textcnn import TextCNN 
from classify.bilstm_attention import BiLSTM_Attention

# 2. text match task model;
from text_match.semantic.DSSM import DSSM
from text_match.semantic.ESIM import ESIM
# from text_match.semantic.BIMPM import BIMPM
# from text_match.semantic.ABCnn import ABCnn

# 3. chatbot/nlg task model;
from chatbot.Seq2SeqWithAtt import Seq2SeqWithAtt

# Train
from train import Train

# Infer
from infer import Infer


argparser = argparse.ArgumentParser()
argparser.add_argument("-t", "--task", default="classify",
                       help="task name: [ classify | ner | text_match ]")
argparser.add_argument("-m", "--model", default="textcnn",
                       help="model type")
argparser.add_argument("-d", "--mode", default="train",
                       help="process mode: [ train | eval ]")
args = argparser.parse_args()

FLAGS = TFlags(args.task, args.model, args.mode).FLAGS

# data processor
# chatbot data_processor
data_path = FLAGS.data_path if args.mode=="train" else FLAGS.eval_data_path

if args.task == 'text_match':
    data_processor = DataProcessor(data_path, 
                                   ftype=3,
                                   maxlen=FLAGS.seq_len,
                                   vpath=FLAGS.vocab_path,
                                   slabel='\t')

data_processor.load_data(args.mode)

## split_data: {'train':['x', 'y'], 
#               'eval' :['x', 'y'], 
#               'test' :['x', 'y']}
if args.mode=='train':
    splited_data = data_processor.data_split(eval=0.01, test=0.01)
    train_data = splited_data['train']
    eval_data  = splited_data['eval']
    test_data  = splited_data['test']
    field_len = len(train_data)
elif args.mode=="eval":
    # doc2ids is `dict` type;
    eval_data = []
    doc2ids = data_processor.doc2ids
    for k,v in doc2ids.items():
        eval_data.append(v)
    field_len = len(eval_data)

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
                model = ESIM(FLAGS.seq_len,
                             FLAGS.num_classes,
                             FLAGS.embedding_dim,
                             len(data_processor.char2idx),
                             FLAGS.hidden_size,
                             FLAGS.learning_rate)
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


        if args.mode == "train":
            trainer = Train(model, FLAGS, sess_config, field_len=field_len)
            trainer.train(sess, train_data, eval_data, test_data)
        elif args.mode == "eval":
            inferor = Infer(model, FLAGS, sess)
            inferor.infer(eval_data, field_len=field_len)
