#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：FFM.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月23日
#   描    述：
#            reference: https://blog.csdn.net/John_xyz/article/details/78933253#%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0
#
#================================================================


class FFM(object):
    """
    Field-aware Factorization Machine
    """
    def __init__(self, config):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.k = config['k']
        # num of fields
        self.f = config['f']
        # num of features
        self.p = feature_length
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reg_l1 = config['reg_l1']
        self.reg_l2 = config['reg_l2']
        self.feature2field = config['feature2field']

    def add_placeholders(self):
        self.X = tf.placeholder('float32', [self.batch_size, self.p])
        self.y = tf.placeholder('int64', [None,])
        self.keep_prob = tf.placeholder('float32')

    def inference(self):
        """
        forward propagation
        :return: labels for each sample
        """
        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[2],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.p, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.matmul(self.X, w1), b)

        with tf.variable_scope('field_aware_interaction_layer'):
            v = tf.get_variable('v', shape=[self.p, self.f, self.k], dtype='float32',
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.field_aware_interaction_terms = tf.constant(0, dtype='float32')
            # build dict to find f, key of feature,value of field
            for i in range(self.p):
                for j in range(i+1,self.p):
                    self.field_aware_interaction_terms += tf.multiply(
                        tf.reduce_sum(tf.multiply(v[i,self.feature2field[i]], v[j,self.feature2field[j]])),
                        tf.multiply(self.X[:,i], self.X[:,j])
                    )
        # shape of [None, 2]
        self.y_out = tf.add(self.linear_terms, self.field_aware_interaction_terms)
        self.y_out_prob = tf.nn.softmax(self.y_out)

    def add_loss(self):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
            mean_loss = tf.reduce_mean(cross_entropy)
            self.loss = mean_loss
            tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        # accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(model.y_out,1), tf.int64), model.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # add summary to accuracy
        tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        # Applies exponential decay to learning rate
        self.global_step = tf.Variable(0, trainable=False)
        # define optimizer
        optimizer = tf.train.AdagradDAOptimizer(self.lr, global_step=self.global_step)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        """build graph for model"""
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()

