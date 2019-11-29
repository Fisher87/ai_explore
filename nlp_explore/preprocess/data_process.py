#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：data_process.py
#   创 建 者：YuLianghua
#   创建日期：2019年11月27日
#   描    述：
#
#================================================================

import codecs
from collections import defaultdict
from collections import Counter

class DataProcessor(object):
    def __init__(self, fpath):
        self.fpath = fpath
        self.text2id = dict()
        self.index2word = dict()

    def load_data(self, fields=None, seg_label='\t'):
        """
        load data from dataset. 
        @param: `fields` means chose which index fields to process, type [index list];
        @param: `seg_label`
        """
        content = ""
        with open(self.fpath, 'r') as rf:
            for line in rf:
                if not fields:
                    content += line.strip()
                    continue
                items = line.strip().split(seg_label)
                get_items = [items[i] for i in fields]
                content += ''.join(get_items)
        return content


    @staticmethod
    def build_dict(content, max_length=None, min_freq=1, keep_back=None):
        """
        @param
        """
        # keep max length content, delete tail;
        if max_length and max_length<len(content):
            content = content[:max_length]

        word_count = Counter(content)
        word_count = sorted(word_count.items(), key=lambda x:x[1], reverse=True)

        # delete word count less then min_freq;
        if min_freq > 0:
            del_index_bound = len(word_count)
            for i, wc in enumerate(word_count):
                w, c = wc
                if c < min_freq:
                    del_index_bound = i
                    break
            word_count = word_count[:del_index_bound]

        # keep back max number;
        if keep_back and keep_back<len(word_count)-1:
            word_count = word_count[:len(word_count)-1]

        # vocab_dict={"<PAD>":0, "<UNK>":1}
        vocab = ["<PAD>", "<UNK>"]
        for w,c in word_count:
            # vocab_dict[w] = len(vocab_dict)
            vocab.append(w)

        return vocab

    @staticmethod
    def gen_text2id(text, vocab_dict, padding=False, max_len=None):
        text2id = []
        for w in text:
            if w in vocab_dict:
                text2id.append(vocab_dict[w])
            else:
                text2id.append(vocab_dict["<UNK>"])
        if padding and max_len is not None:
            text2id += [vocab_dict["<PAD>"]] * (max_len-len(text2id))

        return text2id


if __name__ == "__main__":
    # with open('../data/classify/data.csv') as rf:
    #     content = ""
    #     for line in rf:
    #         content += line.strip().split('\t')[-1]
    fpath = '../data/classify/data.csv'
    processor = DataProcessor(fpath)
    content = processor.load_data(fields=[1])
    vocab = processor.build_dict(content, min_freq=2)
    print(len(vocab))
    with codecs.open('./vocab.txt', 'w', 'utf8') as wf:
        wf.write('\n'.join(vocab))



