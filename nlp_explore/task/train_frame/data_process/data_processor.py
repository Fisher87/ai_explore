#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：data_processor.py
#   创 建 者：YuLianghua
#   创建日期：2020年04月17日
#   描    述：
#
#================================================================

import pdb
import numpy as np

from collections import Counter
from collections import defaultdict


class DataProcessor(object):
    def __init__(self, dpath, 
                 ftype=1, 
                 maxlen=100, 
                 slabel=',',
                 vpath=None,
                 startline=0,
                 padding=True,
                 pvalue = 0,
                 label_one_hot=True,
                 s2_label=False,
                 s2_label_dict = None
                 ):
        '''
        @param dpath:
        @param ftype: int,  1:`<s, y>`, 2:`<s, s>`, 3:`<s, s, y>`
        @param maxlen: sequence max len; default 50;
        @param slabel: split label; default `\t`;
        @param vpath: vocab path;
        @param startline:
        @param padding: boolean, whether to padding; default true;
        @param pvalue:
        @param label_one_hot: boolean,
        @param s2_label: boolean,
        @param s2_label_dict: dict, label2id dict, like, {"B":0, "I":1, "O":2}
        '''
        self.dpath = dpath
        self.ftype = ftype
        self.maxlen = maxlen
        self.slabel = slabel
        self.vpath  = vpath
        self.startline = startline
        self.padding = padding
        self.pvalue = pvalue
        self.label_one_hot = label_one_hot
        self.s2_label = s2_label
        self.s2_label_dict = s2_label_dict
        if not vpath:
            self._init_vocab_dict()
        else:
            char_list = [c.strip() for c in open(self.vpath, 'r').readlines() if c]
            self.char2idx = {c:idx for (idx, c) in enumerate(char_list)}
            self.idx2char = {idx:c for (idx, c) in enumerate(char_list)}
        self.data_list = defaultdict(list)

    def _init_vocab_dict(self):
        print("start to init vocab: %s" %self.dpath)
        with open(self.dpath, 'r') as rf:
            if self.ftype==1:
                content = ''.join([l.strip().split(self.slabel)[0] for l in rf.readlines()])
            elif self.ftype==2:
                if s2_label:
                    content = ''.join([l.strip().split(self.slabel)[0] for l in rf.readlines()])
                else:
                    content = ''.join([l.strip().replace('\t', '') for l in rf.readlines()])
            elif self.ftype==3:
                content = ''.join([l.strip().rsplit(self.label, 1)[0] for l in rf.readlines()])
            char_count = Counter(content)
            char_list = [c[0] for c in char_count.most_common(4000)]
            char_list = ['<PAD>', '<UNK'] + char_list
            self.char2idx = {c:idx for (c, idx) in enumerate(char_list)}
            self.idx2char = {idx:c for (c, idx) in enumerate(char_list)}
            # TODO
            ## save char sets

    def load_data(self):
        '''load data
        '''
        if self.ftype==1:
            fields = ['s1', 'label']
        elif self.ftype==2:
            fields = ['s1', 's2']
        elif self.ftype==3:
            fields = ['s1', 's2', 'label']
        else:
            raise ValueError()
        with open(self.dpath, 'r') as rf:
            for li, line in enumerate(rf):
                if li<self.startline:
                    continue
                items = line.strip().split(self.slabel)
                assert len(items)==len(fields)
                for (f,seq) in zip(fields, items):
                    self.data_list[f].append(seq)
        self._doc2ids()

    def _doc2ids(self):
        self.doc2ids = dict()
        if self.ftype==1 or self.ftype==3:
            for f,seqs in self.data_list.items():
                seq2ids = []
                if f=="label":
                    for seq in seqs:
                        seq2ids.append(int(seq.strip()))
                    if self.label_one_hot:
                        a = np.array(seq2ids)
                        b = np.zeros((a.size, a.max()+1))
                        b[np.arange(a.size),a] = 1
                        seq2ids = b
                else:
                    for seq in seqs:
                        # <UNK> : 1 
                        seq2id = [self.char2idx.get(c, 1) for c in seq]
                        if self.padding:
                            seq2id = padding(seq2id, maxlen=self.maxlen)
                        seq2ids.append(np.array(seq2id))
                self.doc2ids[f] = np.array(seq2ids)
        elif self.ftype==2:
            for f,seqs in self.data_list.items():
                seq2ids = []
                if f=="s2" and self.s2_label:
                    for seq in seqs:
                        seq2id = [self.s2_label_dict[seq.strip()] for c in seq]
                        if self.padding:
                            seq2id = padding(seq2id, maxlen=self.maxlen)
                        seq2ids.append(np.array(seq2id))
                else:
                    for seq in seqs:
                        # <UNK> : 1 
                        seq2id = [self.char2idx.get(c, 1) for c in seq]
                        if self.padding:
                            seq2id = padding(seq2id, maxlen=self.maxlen)
                        seq2ids.append(np.array(seq2id))
                self.doc2ids[f] = np.array(seq2ids)

        del self.data_list

    def _split_index(self, x, eval_rate, test_rate):
        train_rate = 1.0-(eval_rate+test_rate)
        eval_sample_index = int(train_rate * float(len(x)))
        test_sample_index= int((train_rate + eval_rate) * float(len(x)))
        return eval_sample_index, test_sample_index

    def data_split(self, shuffle=True, eval=0.1, test=0.1):
        d = None
        if self.ftype==1:
            s = self.doc2ids['s1']
            y = self.doc2ids['label']
            if shuffle:
                np.random.seed(10)
                shuffle_indices = np.random.permutation(np.arange(len(y)))
                s_shuffled = s[shuffle_indices]
                y_shuffled = y[shuffle_indices]
            s = s_shuffled if shuffle else s
            y = y_shuffled if shuffle else y
            d = (s,y)
        elif self.ftype==2:
            s = self.doc2ids['s1']
            y = self.doc2ids['s2']
            if shuffle:
                np.random.seed(10)
                shuffle_indices = np.random.permutation(np.arange(len(y)))
                s_shuffled = s[shuffle_indices]
                y_shuffled = y[shuffle_indices]
            s = s_shuffled if shuffle else s
            y = y_shuffled if shuffle else y
            d = (s,y)
        elif self.ftype==3:
            s1 = self.doc2ids['s1']
            s2 = self.doc2ids['s2']
            y = self.doc2ids['label']
            if shuffle:
                np.random.seed(10)
                shuffle_indices = np.random.permutation(np.arange(len(y)))
                s1_shuffled = s1[shuffle_indices]
                s2_shuffled = s2[shuffle_indices]
                y_shuffled = y[shuffle_indices]
            s1 = s1_shuffled if shuffle else s1
            s2 = s2_shuffled if shuffle else s2
            y = y_shuffled if shuffle else y
            d = (s1, s2, y)
        else:
            raise ValueError()
        self.split_data = defaultdict(list)
        eval_split_index, test_split_index = self._split_index(d[-1], eval, test)
        for l in d:
            train, eval, test = l[:eval_split_index], \
                                l[eval_split_index : test_split_index], \
                                l[test_split_index:]
            self.split_data['train'].append(train)
            self.split_data['eval'].append(eval)
            self.split_data['test'].append(test)
            
        return self.split_data

def batch_iter(data, batch_size, epoch_num, padding=True, shuffle=True):
    '''generate a batch iterator for dataset;
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size)+1
    for epoch in range(epoch_num):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*batch_size
            end_index = min(batch_size*(batch_num+1), data_size)
            yield shuffled_data[start_index:end_index]

def padding(x, maxlen=50, pvalue=0):
    x = x[:maxlen] if len(x)>=maxlen else x + [0]*(maxlen-len(x))
    return x
