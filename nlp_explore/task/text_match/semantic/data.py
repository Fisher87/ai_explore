#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：data.py
#   创 建 者：YuLianghua
#   创建日期：2020年03月14日
#   描    述：
#
#================================================================

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class DataHandler(object):
    def __init__(self, vocab_path, 
                 max_char_length=None):
        self.vocab_path = vocab_path
        self.load_char_vocab()
        self.max_char_len = max_char_length

    def load_char_vocab(self):
        vocab = [line.strip() for line in open(self.vocab_path, 'r').readlines()]
        self.vocab = vocab
        self.char2index = {char : index for index, char in enumerate(vocab)}
        self.index2char = {index : char for index, char in enumerate(vocab)}
   
    def _sentence_index(self, sentence):
       # "UNK" : 1
       return [self.char2index.get(c.lower(), 1) for c in sentence if len(c.strip())>0]

    def sentences_index(self, seqs):
        p_list = []
        for s in seqs:
            s2i = self._sentence_index(s)
            p_list.append(s2i)

        p_list = self.padding(p_list)

        return p_list

    def padding(self, seqs, padding_value=0):
        lengths = [len(s) for s in seqs]
        sample_len = len(seqs)
        if self.max_char_len is None:
            self.max_char_len = max(lengths)

        padding_list = []
        for l in seqs:
            if len(l) < self.max_char_len:
                # padding value
                l += [padding_value] * (self.max_char_len-len(l))
            else:
                l = l[:self.max_char_len]
            padding_list.append(l)

        return padding_list

    def load_data(self, fpath):
        df = pd.read_csv(fpath)
        df = shuffle(df)
        p = df['sentence1'].values
        h = df['sentence2'].values
        label = df['label'].values

        p_list = self.sentences_index(p)
        h_list = self.sentences_index(h)

        return  p_list, h_list, label

    def batch_iter(self, data, batch_size, num_epochs, 
                   shuffle=True):
        """
        gengrate a batch iterator for dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((data_size-1)/batch_size)+1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num*batch_size
                end_index = min(batch_size*(batch_num+1), data_size)
                yield shuffled_data[start_index:end_index]

