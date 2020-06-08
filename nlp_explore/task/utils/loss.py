#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：loss.py
#   创 建 者：YuLianghua
#   创建日期：2020年06月07日
#   描    述：
#
#================================================================
import tensorflow as tf

# contrastive loss
def contrastive_loss(y, distance, batch_size,
                     scope='contrastive_loss'):
    '''
    Args:
        
    des:
        这种损失函数可以很好的表达成对样本的匹配程度, 也能够很好用于训练提取特征的模型。
        当y=1(即样本相似)时,损失函数只剩下 `tmp`;
        当y=0(即样本不相似)时,损失函数为 `tmp2`;
    '''
    with tf.variable_scope(scope=scope, reuse=tf.AUTO_REUSE):
        tmp = y * tf.square(distance)
        tmp2 = (1-y) * tf.square(tf.maxmium(1-distance, 0))
        return tf.reduce_sum(tmp +tmp2)/tf.cast(batch_size,tf.float32)/2

