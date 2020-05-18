#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：train_frame.py
#   创 建 者：YuLianghua
#   创建日期：2020年04月19日
#   描    述：NLP 训练通用框架
#
#================================================================

import os
import datetime
import tensorflow as tf

from data_process.data_processor import batch_iter 

class TrainBaseFrame(object):
    def __init__(self, model, flags, sess_config):
        self.model = model
        self.flags = flags
        self.sess_config = sess_config
        self.best_loss = float('inf')
        self.global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")

    def init_summaries(self, sess):
        ''' keep the track of gradient values and sparsity
        '''
        grad_summaries = []
        grads_and_vars = self.model.grads_and_vars
        for g,v in grads_and_vars:
            if g is not None:
                # gradient value
                grad_hist_summary=tf.compat.v1.summary.histogram("{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)
                # sparity
                sparsity_summary =tf.compat.v1.summary.scalar('{}/grad/sparsity'.format(v.name),
                                                              tf.nn.zero_fraction(g))
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.compat.v1.summary.merge(grad_summaries)

        # summaries for loss and accuracy
        loss_summary=tf.compat.v1.summary.scalar('loss', self.model.loss)
        accuracy_summary=tf.compat.v1.summary.scalar('accuracy', self.model.acc)

        # train summaries
        self.train_summary_op = tf.compat.v1.summary.merge([loss_summary,accuracy_summary,
                                                            grad_summaries_merged])
        train_summary_dir=os.path.join(self.flags.out_dir,'summaries','train')
        self.train_summary_writer=tf.compat.v1.summary.FileWriter(train_summary_dir,sess.graph)

        # dev summaries
        self.dev_summary_op = tf.compat.v1.summary.merge([loss_summary, accuracy_summary])
        dev_summary_dir=os.path.join(self.flags.out_dir,'summaries','dev')
        self.dev_summary_writer=tf.compat.v1.summary.FileWriter(dev_summary_dir,sess.graph)

    def init_saver(self):
        checkpoint_dir = os.path.abspath(os.path.join(self.flags.out_dir, 'checkpoint_dir'))
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.flags.num_checkpoints)
        return saver

    def train_step(self, session, feed_dict):
        if self.flags.summaries:
            _, steps, summaries, loss, acc = session.run([self.train_op,
                                                          self.global_step,
                                                          self.train_summary_op,
                                                          self.model.loss,
                                                          self.model.acc], feed_dict=feed_dict)
            nowtime=datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, accuracy {:g}'.format(nowtime,steps,loss,acc))
            self.train_summary_writer.add_summary(summaries,steps)
        else:
            _, steps, loss, acc = session.run([self.train_op,
                                               self.global_step,
                                               self.model.loss,
                                               self.model.acc], feed_dict=feed_dict)
            nowtime=datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, accuracy {:g}'.format(nowtime,steps,loss,acc))

    def eval_step(self, session, feed_dict):
        if self.flags.summaries:
            steps, summaries, loss, acc = session.run([self.global_step,
                                                       self.dev_summary_op,
                                                       self.model.loss,
                                                       self.model.acc], feed_dict=feed_dict)
            nowtime=datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, accuracy {:g}'.format(nowtime,steps,loss,acc))
            self.dev_summary_writer.add_summary(summaries,steps)
        else:
            steps, loss, acc = session.run([self.global_step,
                                            self.model.loss,
                                            self.model.acc], feed_dict=feed_dict)
            nowtime=datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, accuracy {:g}'.format(nowtime,steps,loss,acc))

        return loss
        
    def early_stopping(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.last_improvement = 0
        else:
            self.last_improvement += 1

        if self.last_improvement >= 20:
            return True
        else:
            return False

    def get_batches(self, train_data, batch_size, num_epochs, shuffle=True):
        '''
        if train_data: ([train_x], [train_y])
        >>> data = list(zip(train_data))
        >>> batches = batch_iter(data, batch_size, num_epochs, shuffle=shuffle)
        >>> batches
        '''
        raise NotImplementedError

    def get_feed_dict(self, batch_data, is_training=True):
        '''
        if batch_data is iteration like: (train_x, train_y)
        >>> x_batch, y_batch = zip(*batch)
        >>> feed_dict = {self.model.input_x:x_batch, self.model.input_y:y_batch, \
        >>>                 self.model.dropout_keep_prob:self.flags.dropout_keep_prob}
        '''
        raise NotImplementedError

    def train(self, sess, train_data, eval_data, test_data):
        if self.flags.summaries:
            self.init_summaries(sess)

        self.train_op = self.model.optimizer.apply_gradients(
                                            self.model.grads_and_vars,
                                            global_step=self.global_step)

        saver = self.init_saver()
        sess.run(tf.global_variables_initializer())

        batches = self.get_batches(train_data, self.flags.batch_size, 
                                  self.flags.num_epochs)
        for batch in batches:
            feed_dict = self.get_feed_dict(batch, is_training=True)
            self.train_step(sess, feed_dict)
            current_step = tf.train.global_step(sess, self.global_step)

            if current_step % self.flags.evaluate_every == 0:
                feed_dict = self.get_feed_dict(eval_data)
                print('\nEvaluation!')
                current_loss = self.eval_step(sess, feed_dict)
                print('\n')

                # check early stopping
                if self.early_stopping(current_loss):
                    print("Early stopping is trigger at step: {} loss:{}".\
                          format(current_step,current_loss))
                    path = saver.save(sess, self.checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    break

            if current_step % self.flags.checkpoint_every == 0:
                path = saver.save(sess, self.checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        # finally test
        feed_dict = self.get_feed_dict(test_data)
        print('\nTest Finally!')
        self.eval_step(sess, feed_dict)

    def infer(self, sess, infer_data):
        # TODO
        pass
