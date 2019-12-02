#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：train.py
#   创 建 者：YuLianghua
#   创建日期：2019年11月28日
#   描    述：
#
#================================================================
import os, sys
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-3])
sys.path.append(ROOT_PATH)

import time
import pdb
import logging
import datetime
import tensorflow as tf

from utils.data_helper import DataHelper
from utils.data_helper import train_test_data_split
from utils.data_helper import batch_iter
from utils.data_helper import padding
from utils.data_helper import one_hot_encode
from utils.logger import glogger
from textcnn import TextCNN

# 1. init_flags
TRAIN_MODEL = input("Select train model.[textcnn, ...]: ")
while not (TRAIN_MODEL.isalpha() and TRAIN_MODEL.lower() in ["textcnn"]):
    TRAIN_MODEL = input("✘ Model select is illegal, please re-input: ")

TRAIN_OR_RESTORE = input("Train or Restore?(T/R): ")
while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("✘ The format of your input is illegal, please re-input: ")
logging.info("✔︎ The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = glogger("tflog", "logs/training-{0}.log".format(time.asctime()))
if TRAIN_OR_RESTORE == 'R':
    logger = glogger("tflog", "logs/restore-{0}.log".format(time.asctime()))

## data parameters
data_fpath = "../../data/classify/data.csv"
vocab_fpath = "../../data/vocab.txt"
metadata_fpath = ""
out_dir = "./result/"

tf.flags.DEFINE_string("data_path", data_fpath, "data source file path.")
tf.flags.DEFINE_string("vocab_path", vocab_fpath, "vocab data source file path.")
tf.flags.DEFINE_string("metadata_path", metadata_fpath, "metadata file for embedding visualization"
                                                        "(each line is a word segment in metadata_file).")
tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "train or restore.")

## model hyparameters
### textcnn
if TRAIN_MODEL == "textcnn":
    tf.flags.DEFINE_integer("pad_seq_len", 100, "padding seq length of data.")
    tf.flags.DEFINE_integer("embedding_dim", 100, "deminsionality of embedding size. (default: 128)")
    tf.flags.DEFINE_integer("embedding_type", 1, "the embedding type. (default: 1)")
    tf.flags.DEFINE_float("learning_rate", 0.001, "the learning rate. (default: 0.001)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "filter sizes. (default: 3,4,5)")
    tf.flags.DEFINE_integer("num_filters", 128, "number filters output for per filter size. (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "l2 regularization lambda. (default: 0.0)")
    tf.flags.DEFINE_integer("fc_hidden_size", 1024, "hidden size for fully connected layer (default: 1024).")
    tf.flags.DEFINE_integer("num_classes", 2, "number of labels (depends on the task.)")
    tf.flags.DEFINE_integer("top_num", 5, "number of top K prediction classes. (default: 5)")
    tf.flags.DEFINE_float("threshold", 0.5, "threshold for prediction classes. (default: 0.5)")
elif TRAIN_MODEL == "textrnn":
    pass

## training parameters
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

## misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "allow gpu options growth")

FLAGS=tf.flags.FLAGS
print("parameters info:")
for attr, value in tf.flags.FLAGS.__flags.items():
    print("{0}: {1}".format(attr, value.value))

# 2. load data
## get train_x, train_y, dev_x, dev_y
data_helper = DataHelper(FLAGS.data_path, FLAGS.vocab_path, fields=['y', 'x'], startline=1)
data_list = data_helper.get_data(id_fields=['x'])
x, y = data_list['x'], data_list['y']
padding_x, max_document_length = padding(x, maxlen=FLAGS.pad_seq_len)
int_y = [int(_y) for _y in y]
encoded_y = one_hot_encode(int_y) 
train_x, test_x, train_y, test_y = train_test_data_split(padding_x, encoded_y)

# 3. define session
with tf.Graph().as_default():
    # session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    # sess=tf.Session(config=session_config)
    session_config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess=tf.compat.v1.Session(config=session_config)
    with sess.as_default():
        model = TextCNN(FLAGS.pad_seq_len, 
                        FLAGS.num_classes, 
                        len(data_helper.token2idx), 
                        FLAGS.embedding_dim, 
                        FLAGS.learning_rate,
                        FLAGS.filter_sizes,
                        FLAGS.num_filters,
                        FLAGS.random_embedding,
                        FLAGS.l2_reg_lambda)
        
        # 4. define important variable
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        # 5. record `summaries`,like:scalars, graph, histogram
        ## I. keep the track of gradient values and sparsity
        grad_summaries = []
        for g,v in grads_and_vars:
            if g is not None:
                # gradient value
                grad_hist_summary=tf.compat.v1.summary.histogram('{}/grad/hist'.format(v.name),g)
                grad_summaries.append(grad_hist_summary)
                # sparsity
                sparsity_summary =tf.compat.v1.summary.scalar('{}/grad/sparsity'.format(v.name),tf.nn.zero_fraction(g))
                grad_summaries.append(sparsity_summary)
                grad_summaries_merged=tf.compat.v1.summary.merge(grad_summaries)

        ## II. summaries for loss and accuracy
        loss_summary=tf.compat.v1.summary.scalar('loss', model.loss)
        accuracy_summary=tf.compat.v1.summary.scalar('accuracy', model.accuracy)

        ## III. train summaries
        train_summary_op = tf.compat.v1.summary.merge([loss_summary,accuracy_summary,grad_summaries_merged])
        train_summary_dir=os.path.join(FLAGS.out_dir,'summaries','train')
        train_summary_writer=tf.compat.v1.summary.FileWriter(train_summary_dir,sess.graph)

        ## IV. dev summaries
        dev_summary_op = tf.compat.v1.summary.merge([loss_summary, accuracy_summary])
        dev_summary_dir=os.path.join(FLAGS.out_dir,'summaries','dev')
        dev_summary_writer=tf.compat.v1.summary.FileWriter(dev_summary_dir,sess.graph)

        ## V. checkpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "checkpoint_dir"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # 6. initialize all variable
        sess.run(tf.global_variables_initializer())

        # 7. train 
        ## I. set train step func.
        def train_step(x_batch, y_batch):
            feed_dict={model.input_x:x_batch,
                       model.input_y:y_batch,
                       model.dropout_keep_prob: FLAGS.dropout_keep_prob}
            _,step,summaries,loss,accuracy=sess.run([train_op, 
                                                     global_step,
                                                     train_summary_op,
                                                     model.loss,
                                                     model.accuracy],
                                                         feed_dict=feed_dict)
            nowtime=datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, accuracy {:g}'.format(nowtime,step,loss,accuracy))
            train_summary_writer.add_summary(summaries,step)
        ## II. set evl step fun.
        def evl_step():
            feed_dict = {model.input_x: x_batch,
                         model.input_y: y_batch,
                         model.dropout_keep_prob: FLAGS.dropout_keep_prob}

            step, summaries, loss, accuracy = sess.run([global_step, 
                                                        dev_summary_op, 
                                                        model.loss, 
                                                        model.accuracy],
                                                           feed_dict=feed_dict)
            nowtime = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, accuracy {:g}'.format(nowtime, step, loss, accuracy))
            dev_summary_writer.add_summary(summaries, step)

        batches = batch_iter(list(zip(train_x,train_y)), 
                                        FLAGS.batch_size,
                                        num_epochs=FLAGS.num_epochs,
                                        shuffle=True)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print('\nEvaluation!')
                dev_step(x_batch, y_batch)
                print('\n')

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

            # 8. early stop.

def preprocess(data_helper):
    vocab = data_helper.load_vocab()
    data_list = data_helper.get_data(id_fields=['x'])
    x, y = data_list['x'], data_list['y']
    x, max_document_length = data_helper.padding(x, maxlen=50)
    all_data = zip(x, y)
    x_train, y_train, x_test, y_test = train_test_data_split(x, y)
    return x_train, y_train, x_test, y_test

# if __name__ == "__main__":
#     fpath = "../../data/classify/data.csv"
#     vocab_path = "../../data/vocab.txt"
#     data_helper = DataHelper(fpath, vocab_path, fields=['y', 'x'], startline=1)
#     # x_train, y_train, x_dev, y_dev = preprocess(data_helper)
#     preprocess(data_helper)




