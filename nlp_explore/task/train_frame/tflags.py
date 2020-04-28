#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：flags.py
#   创 建 者：YuLianghua
#   创建日期：2020年04月20日
#   描    述：
#
#================================================================

import tensorflow as tf

data_fpath = "./data/data.csv"
vocab_fpath = "./data/vocab.txt"
metadata_fpath = ""
out_dir = "./result/"

class TFlags(object):
    def __init__(self, task, model):
        # system flags
        tf.flags.DEFINE_string("data_path", data_fpath, "data source file path.")
        tf.flags.DEFINE_string("vocab_path", vocab_fpath, "vocab data source file path.")
        tf.flags.DEFINE_string("metadata_path", metadata_fpath, "metadata file for embedding visualization"
                                                        "(each line is a word segment in metadata_file).")
        # training parameters
        tf.flags.DEFINE_boolean("summaries", True, "using summaries")
        tf.flags.DEFINE_boolean("random_embedding", True, "random generate embedding table.(default:True)")
        tf.flags.DEFINE_integer("batch_size", 1024, "batch size (default:256)")
        tf.flags.DEFINE_integer("num_epochs", 150, "number of training epoch. (default: 100)")
        tf.flags.DEFINE_integer("evaluate_every", 10, "evalutate model on dev set after this many steps. (default:5000)")
        tf.flags.DEFINE_float("norm_ratio", 2, "the ratio of the sum of gradients norms of trainable vaiable. (defalut:1.25)")
        tf.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate. (default: 500)")
        tf.flags.DEFINE_float("decay_rate", 0.95, "rate of decay for learning rate. (default: 0.95)")
        tf.flags.DEFINE_integer("checkpoint_every", 10, "save model after this many steps (default: 1000)")
        tf.flags.DEFINE_string("out_dir", out_dir, "save training data (like:model, checkpoint, summaries) path")
        tf.flags.DEFINE_integer("num_checkpoints", 10, "number of checkpoints to store (default: 50)")
        tf.flags.DEFINE_integer("early_stopping_span", 25, "number of steps span that eval loss not descend (default: 25)")

        ## misc parameters
        tf.flags.DEFINE_boolean("allow_soft_placement", True, "allow device soft device placement")
        tf.flags.DEFINE_boolean("log_device_placement", False, "log placement of ops on devices")
        tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "allow gpu options growth")

        # task model parameters
        if task=='classify':
            if model=='textcnn':
                tf.flags.DEFINE_integer("pad_seq_len", 100, "padding seq length of data.")
                tf.flags.DEFINE_integer("embedding_dim", 100, "deminsionality of embedding size. (default: 128)")
                tf.flags.DEFINE_float("learning_rate", 0.001, "the learning rate. (default: 0.001)")
                tf.flags.DEFINE_string("filter_sizes", "3,4,5", "filter sizes. (default: 3,4,5)")
                tf.flags.DEFINE_integer("num_filters", 128, "number filters output for per filter size. (default: 128)")
                tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout keep probability (default: 0.5)")
                tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "l2 regularization lambda. (default: 0.0)")
                tf.flags.DEFINE_integer("fc_hidden_size", 1024, "hidden size for fully connected layer (default: 1024).")
                tf.flags.DEFINE_integer("num_classes", 2, "number of labels (depends on the task.)")
                tf.flags.DEFINE_integer("top_num", 5, "number of top K prediction classes. (default: 5)")
                tf.flags.DEFINE_float("threshold", 0.5, "threshold for prediction classes. (default: 0.5)")

            elif model=='bilstm_attention':
                tf.flags.DEFINE_integer("seq_len", 100, "padding seq length of data.")
                tf.flags.DEFINE_integer("embedding_dim", 100, "deminsionality of embedding size. (default: 128)")
                tf.flags.DEFINE_integer("hidden_size", 128, "lstm hidden layer units. (default: 128)")
                tf.flags.DEFINE_float("learning_rate", 0.001, "the learning rate. (default: 0.001)")
                tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout keep probability (default: 0.5)")
                tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "l2 regularization lambda. (default: 0.0)")
                tf.flags.DEFINE_integer("attention_size", 64, "attention size (default: 64).")
                tf.flags.DEFINE_integer("num_classes", 2, "number of labels (depends on the task.)")
                tf.flags.DEFINE_integer("top_num", 5, "number of top K prediction classes. (default: 5)")
                tf.flags.DEFINE_float("threshold", 0.5, "threshold for prediction classes. (default: 0.5)")

        elif task=='ner':
            pass

        elif task=='text_match':
            pass

    @property
    def FLAGS(self):
        return tf.flags.FLAGS

