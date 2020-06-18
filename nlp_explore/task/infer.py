#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：infer.py
#   创 建 者：YuLianghua
#   创建日期：2020年06月17日
#   描    述：
#
#================================================================
import os
import pdb
import tensorflow as tf

from train_frame.data_process.data_processor import batch_iter

class Infer(object):
    def __init__(self, model, flags, sess):
        self.model = model
        self.flags = flags
        self.sess = sess
        self._load_model()

    def _load_model(self):
        # self.sess.run(tf.global_variables_initializer())

        checkpoint_dir = os.path.abspath(os.path.join(self.flags.out_dir, \
                                                      'checkpoint_dir'))
        self.saver = tf.train.Saver()
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        # self.saver =  tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        self.saver.restore(self.sess, checkpoint_file)

    def get_batches(self, data, batch_size):
        if len(data)==2:
            data = list(zip(data[0], data[1]))
        batches = batch_iter(data, batch_size, 1, shuffle=False)
        return batches

    def get_feed_dict(self, batch_data, field_len, padding=True, samelen=False):
        if field_len == 1:
            x_batch = list(batch_data)
        elif field_len == 2:
            x_batch, y_batch = zip(*batch_data)

        if padding:
            if field_len == 1:
                x_maxlen = max([len(x) for x in x_batch])
                _x_batch = [list(x)+[0]*(x_maxlen-len(x)) for x in x_batch]

            elif field_len==2:
                x_maxlen = max([len(x) for x in x_batch])
                y_maxlen = max([len(y) for y in y_batch])
                if samelen:
                    x_maxlen = y_maxlen = max(x_maxlen, y_maxlen)
                _x_batch = [list(x)+[0]*(x_maxlen-len(x)) for x in x_batch]
                _y_batch = [list(y)+[0]*(y_maxlen-len(y)) for y in y_batch]
                x_batch = _x_batch
                y_batch = _y_batch

        if field_len == 1:
            feed_dict = {
                self.model.input_x : x_batch,
                self.model.dropout_keep_prob : 1.0
            }

        elif field_len == 2:
            feed_dict = {
                self.model.input_x : x_batch,
                self.model.input_y : y_batch,
                self.model.dropout_keep_prob : 1.0
            }

        return feed_dict
    
    def infer(self, eval_data, field_len=2):
        total_predictions = []
        batches = self.get_batches(eval_data, self.flags.batch_size)
        for batch in batches:
            feed_dict = self.get_feed_dict(batch, field_len)
            predictions = self.sess.run([self.model.predictions], feed_dict=feed_dict)
            total_predictions.extend(predictions)

        print(total_predictions)

