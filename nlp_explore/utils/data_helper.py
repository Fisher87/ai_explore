#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：data_helper.py
#   创 建 者：YuLianghua
#   创建日期：2019年11月28日
#   描    述：
#
#================================================================
import os, sys
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-2])
print(ROOT_PATH)
sys.path.append(ROOT_PATH)

import numpy as np
# import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import train_test_split


class DataHelper(object):
    def __init__(self, fpath, vocab_path, seg_label='\t', fields=[], startline=0):
        self.fpath = fpath
        self.vocab_path = vocab_path
        self.seg_label = seg_label
        self.fields = fields
        self.startline = startline
        self.token2idx = dict()

    def load_data(self, fields=[], seg_label='\t', startline=0):
        """
        get data from `fpath`.
        @fpath:

        return data_list: type dict(list)
        """
        data_list = defaultdict(list)
        with open(self.fpath, 'r') as f:
            for li, line in enumerate(f):
                if li<startline:
                    continue
                items = line.strip().split(seg_label)
                if not fields:
                    fields = [str(i) for i in range(len(items))]
                else:
                    assert len(items)==len(fields)
                for (f, c) in zip(fields, items):
                    data_list[f].append(c)

        return data_list
                    
    def load_vocab(self):
        """

        """
        vocab = [line.strip() for line in open(self.vocab_path, 'r').readlines()]
        token2idx = {token:idx for idx, token in enumerate(vocab)}
        idx2token = {idx:token for idx, token in enumerate(vocab)}

        return token2idx, idx2token 

    def get_data(self, id_fields=["x"]):
        """
        fit `x:label` data, like classify task.
        
        return data_list
        """
        self.data_list = self.load_data(fields   =self.fields, 
                                       seg_label=self.seg_label, 
                                       startline=self.startline)
        self.token2idx, self.idx2token = self.load_vocab() 
        
        for field in id_fields:
            assert field in self.data_list, ("{0} not in data_list".format(field))
            data = self.data_list[field]
            idlist = []
            for text in data:
                l = [int(self.token2idx.get(t, "1")) for t in text]
                idlist.append(l)
            self.data_list[field] = idlist

        return self.data_list

def padding(data, maxlen=None):
    if maxlen is None:
        maxlen = max([len(d) for d in data])

    idlist = []
    for l in data:
        if len(l)<maxlen:
            # `padding:0, UNK:1`
            l += [0] * (maxlen-len(l))
        idlist.append(l)

    return (idlist, maxlen)

def train_test_data_split(data, shuffle=True, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle)

    return (x_train, x_test, y_train, y_test)

def train_dev_test_data_split(data, shuffle=True, dev_size=0.1, test_size=0.2):
    _x_data = [d[0] for d in data]
    _y_data = [d[1] for d in data]
    if shuffle:
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(_y_data)))
        x_shuffled = _x_data[shuffle_indices]
        y_shuffled = _y_data[shuffle_indices]
    x_data = x_shuffled if shuffle else _x_data
    y_data = y_shuffled if shuffle else _y_data

    train_size = 1.0 - (dev_size + test_size)
    dev_sample_index = int(train_size * float(len(_y_data)))
    test_sample_index= int((train_size + dev_size) * float(len(_y_data)))
    x_train, x_dev, x_test = x_data[:dev_sample_index], x_data[dev_sample_index:test_sample_index], x_data[test_sample_index:]
    y_train, y_dev, y_test = y_data[:dev_sample_index], y_data[dev_sample_index:test_sample_index], y_data[test_sample_index:]

    del x_data, y_data, _x_data, _y_data

    return (x_train, y_train, x_dev, y_dev, x_test, y_test)

def batch_iter(data, batch_size, num_epoches, shuffle=True):
    """
    gengrate a batch iterator for dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_pre_epoch = int(data_size-1) / batch_size + 1



if __name__ == "__main__":
    fpath = "../data/classify/data.csv"
    vocab_path = "../data/vocab.txt"
    data_helper = DataHelper(fpath, vocab_path, fields=["y", "x"], startline=1)
    data_list = data_helper.get_data(padding=False)
    print(data_list["x"][:10])
