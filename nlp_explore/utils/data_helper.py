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


class DataHelper(object):
    def __init__(self, fpath, vocab_path, seg_label='\t', fields=[], startline=0):
        self.fpath = fpath
        self.vocab_path = vocab_path
        self.seg_label = seg_label
        self.fields = fields
        self.startline = startline
        self.token2idx = dict()

    @staticmethod
    def load_data(fpath, fields=[], seg_label='\t', startline=0):
        """
        get data from `fpath`.
        @fpath:

        return data_list: type dict(list)
        """
        data_list = defaultdict(list)
        with open(fpath, 'r') as f:
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
                    
    @staticmethod
    def load_vocab(vocab_path):
        """

        """
        vocab = [line.strip() for line in open(vocab_path, 'r').readlines()]
        token2idx = {token:idx for idx, token in enumerate(vocab)}
        idx2token = {idx:token for idx, token in enumerate(vocab)}

        return token2idx, idx2token 

    def get_data(self, id_fields=["x"], padding=True, maxlen=None):
        """
        fit `x:label` data, like classify task.
        
        return data_list
        """
        self.data_list = self.load_data(self.fpath, 
                                   fields   =self.fields, 
                                   seg_label=self.seg_label, 
                                   startline=self.startline)
        self.token2idx, self.idx2token = self.load_vocab(vocab_path) 
        for field in id_fields:
            assert field in self.data_list, ("{0} not in data_list".format(field))
            data = self.data_list[field]
            if padding and maxlen is None:
                maxlen = max([len(d) for d in data])
            idlist = []
            for text in data:
                # `padding:0, UNK:1`
                l = [int(self.token2idx.get(t, "1")) for t in text]
                if padding and len(l)<maxlen:
                    l += [0] * (maxlen-len(l))
                idlist.append(l)
            self.data_list[field] = idlist

        return self.data_list

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
