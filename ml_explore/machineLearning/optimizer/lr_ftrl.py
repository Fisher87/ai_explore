#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：ftrl_lr.py
#   创 建 者：YuLianghua
#   创建日期：2020年12月23日
#   描    述：
#
#================================================================

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import gzip
import random
import json
import argparse


class FTRLProximal(object):
    """
    FTRL Proximal engineer project with logistic regression
    Reference:
    https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/41159.pdf

    """

    def __init__(self, alpha, beta, L1, L2, D,
                 interaction=False, dropout=1.0,
                 dayfeature=True,
                 device_counters=False):

        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        self.dayfeature = dayfeature
        self.device_counters = device_counters

        # feature related parameters
        self.D = D
        self.interaction = interaction
        self.dropout = dropout

        # model
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = [0.] * D

    def _indices(self, x):
        '''
        A helper generator that yields the indices in x
        The purpose of this generator is to make the following
        code a bit cleaner when doing feature interaction.
        '''

        for i in x:
            yield i

        if self.interaction:
            D = self.D
            L = len(x)
            for i in range(1, L):  # skip bias term, so we start at 1
                for j in range(i + 1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x, dropped=None):
        """
        use x and computed weight to get predict
        :param x:
        :param dropped:
        :return:
        """
        # wTx is the inner product of w and x
        wTx = 0.
        for j, i in enumerate(self._indices(x)):

            if dropped is not None and dropped[j]:
                continue

            wTx += self.w[i]

        if dropped is not None:
            wTx /= self.dropout

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, y):
        """
        update weight and coordinate learning rate based on x and y
        :param x:
        :param y:
        :return:
        """

        ind = [i for i in self._indices(x)]

        if self.dropout == 1:
            dropped = None
        else:
            dropped = [random.random() > self.dropout for i in range(0, len(ind))]

        p = self.predict(x, dropped)

        # gradient under logloss
        g = p - y

        # update z and n
        for j, i in enumerate(ind):

            # implement dropout as overfitting prevention
            if dropped is not None and dropped[j]:
                continue

            g_i = g * i
            sigma = (sqrt(self.n[i] + g_i * g_i) - sqrt(self.n[i])) / self.alpha
            self.z[i] += g_i - sigma * self.w[i]
            self.n[i] += g_i * g_i

            sign = -1. if self.z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights -
            if sign * self.z[i] <= self.L1:
                # w[i] vanishes due to L1 regularization
                self.w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get
                self.w[i] = (sign * self.L1 - self.z[i]) \
                            / ((self.beta + sqrt(self.n[i])) / self.alpha + self.L2)

    def save_model(self, save_file):
        """
        保存weight数据到本地
        :param save_file:
        :return:
        """
        with open(save_file, "w") as f:
            w = {k: v for k, v in enumerate(self.w) if v != 0}
            z = {k: v for k, v in enumerate(self.z) if v != 0}
            n = {k: v for k, v in enumerate(self.n) if v != 0}
            data = {
                'w': w,
                'z': z,
                'n': n
            }
            json.dump(data, f)

    def load_weight(self, model_file, D):
        """
        loada weight data
        :param model_file:
        :return:
        """
        with open(model_file, "r") as f:
            data = json.load(f)
            self.w = data.get('w', [0.] * D)
            self.z = data.get('z', [0.] * D)
            self.n = data.get('n', [0.] * D)

    @staticmethod
    def loss(y, y_pred):
        """
        log loss for LR model
        :param y:
        :param y_pred:
        :return:
        """
        p = max(min(y_pred, 1. - 10e-15), 10e-15)
        return -log(p) if y == 1. else -log(1. - p)


def data(f_train, D, dayfilter=None, dayfeature=True, counters=False):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    device_ip_counter = {}
    device_id_counter = {}

    for t, row in enumerate(DictReader(f_train)):
        # process id
        ID = row['id']
        del row['id']

        # process clicks
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        # turn hour really into hour, it was originally YYMMDDHH

        date = row['hour'][0:6]
        row['hour'] = row['hour'][6:]

        if dayfilter != None and not date in dayfilter:
            continue

        if dayfeature:
            # extract date
            row['wd'] = str(int(date) % 7)
            row['wd_hour'] = "%s_%s" % (row['wd'], row['hour'])

        if counters:
            d_ip = row['device_ip']
            d_id = row["device_id"]
            try:
                device_ip_counter[d_ip] += 1
                device_id_counter[d_id] += 1
            except KeyError:
                device_ip_counter[d_ip] = 1
                device_id_counter[d_id] = 1
            row["ipc"] = str(min(device_ip_counter[d_ip], 8))
            row["idc"] = str(min(device_id_counter[d_id], 8))

        # build x
        x = [0]  # 0 is the index of the bias term
        for key in row:
            value = row[key]
            # one-hot encode everything with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append(index)
        yield t, ID, x, y
