#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：lr.py
#   创 建 者：YuLianghua
#   创建日期：2020年07月15日
#   描    述：
#
#================================================================

import numpy as np

class LogicRegression(object):
    def __init__(self, learning_rate, iteration=50, seed=0):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.seed = seed

    def sigmoid(self, x, w):
        '''sigmoid function.'''
        return 1.0 / (1.0 + exp(np.dot(w, x)))

    def sigmoid_gd(self, xs, ys):
        # 梯度上升(求交叉熵最大值)
         sgd = self.learning_rate * (xi * yi - (np.exp(np.dot(self.W, xs)) * xi) / ( 1 + np.exp(np.dot(self.W, xs))))
         return sgd

    def loss(self, xs, ys):
        ''' get loss
        @param Y: label value.
        @param y_hat: predict value.
        '''
        # 1 / N  * \sum (y_i * logy_hat_i + (1-y_i) * log(1-y_hat_i))
        y_hats = self.sigmoid(xs)
        losses = ys * np.log(y_hats) + (1-ys) * log(1-y_hats)
        loss = np.sum(losses) / (len(ys) + 0.0001)
        return loss

    def fit(self, X, Y):
        '''train 
        '''
        feat_nums, data_nums = X.shape
        # W : (D * 1)
        self.W = np.random.uniform(-1.0, 1.0, [feat_nums])
        for i in range(self.iteration):
            # SGD
            for j in range(data_nums):
                loss = self.loss(xs, ys)
                if loss <= self.thresh:
                    break
                self.W += sigmoid_gd(xs, ys) 

    def predict(self, x):
        p = self.sigmoid(x, self.W)
        if p>0.5:
            return 1
        else: return 0
        

