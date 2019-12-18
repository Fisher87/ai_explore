#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：GBDT_LR.py
#   创 建 者：Yulianghua
#   创建日期：2019年11月27日
#   描    述：
#        reference: [1]. https://github.com/neal668/LightGBM-GBDT-LR/blob/master/GBFT%2BLR_simple.py
#
#================================================================

import lightgbm as lgb
import numpy as np
import json
from sklean.linear_model import LogisticRegression

# lgb configurations
num_leaf = 64
num_trees= 100
params={
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': num_leaf,
    'num_trees': num_trees,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

def load_data():
    pass

x_train, x_test, y_train, y_test = load_data()

lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval  = lgb.Dataset(x_test, y_test, reference=lgb_train)

# start train
gbm = lgb.train(params, 
               lgb_train,
               num_boost_round=100,
               valid_sets=lgb_eval,
               early_stopping_rounds=5)
# save model
gbm.save_model("./model")

# 用训练好的lgb模型预测训练集,观察其落在哪些叶子节点上.
y_pred = gbm.predict(x_train, pred_leaf=True)   # [None, num_trees]
# >>> y_pred.shape
# >>> (8000, 100) 
# NOTE: 8000表示样本数为8000, 100表示模型拥有100颗子树
# >>> y_pred[0][:10]
# >>> array([31, 29, 29, 32, 38, 46, 35, 36, 36, 42])
# NOTE: 观察第一个样本的前10个值: 
#    第一个数 31 表示这个样本落到了第一颗树的 31 叶子节点，
#    29 表示落到了第二棵树的 29 叶子节点，
#    注意31 、29表示节点编号，从0开始到63

# feature transformation and write training data
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0])*num_leaf], dtype=np.int32)
# 将叶子节点编号转化为one-hot编码
for i in range(len(y_pred)):
    tmp = np.arange(len(y_pred[0]))*num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][tmp] += 1

y_pred_test = gbm.predict(x_test, pred_leaf=True)
transformed_testing_matrix = np.zeros([len(y_pred_test), len(y_pred_test[0])*num_leaf], dtype=np.int32)
for i in range(len(y_pred_test)):
    tmp = np.arange(len(y_pred_test[0]))*num_leaf + np.array(y_pred_test[i])
    transformed_testing_matrix[i][tmp] += 1


# LR model
cs = np.array([1,0.5,0.1,0.05,0.01,0.005,0.001])
for c in cs:
    lm = LogisticRegression(penalty='l2', C=c)
    lm.fit(transformed_training_matrix, y_train)

    y_pred_test = lm.predict_prob(transformed_testing_matrix)
    NE = (-1) / len(y_pred_test) * sum(((1+y_test)/2 * np.log(y_pred_test[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_test[:,1])))
    print("Normalized Cross Entropy " + str(NE))
