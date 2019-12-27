#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：metrics.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月27日
#   描    述：
#
#================================================================

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

def metrics(labels, logits):
    auc = roc_auc_score(labels, logits)
    loss = log_loss(labels, logits)
    acc = accuracy_score(labels, logits.round())
    precision = precision_score(labels, logits.round())
    recall = recall_score(labels, logits.round())
    f1 = f1_score(labels, logits.round())
    return auc, loss, acc, precision, recall, f1


