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

from collections import defaultdict
from collections import Counter

class DataProcessor(object):
    def __init__(self):
        # self.path = path
        self.text2id = dict()
        self.index2word = dict()

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

        vocab_dict={"<PAD>":0, "<UNK>":1}
        for w,c in word_count:
            vocab_dict[w] = len(vocab_dict)

        return vocab_dict

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
    processor = DataProcessor()
    content = "这些方法是在一个固定的图上直接学习每个节点embedding，但是大多情况图是会演化的，当网络结构改变以及新节点的出现，直推式学习需要重新训练（复杂度高且可能会导致embedding会偏移），很难落地在需要快速生成未知节点embedding的机器学习系统上。"
    vocab = processor.build_dict(content)
    text = "我要听的歌"
    print(vocab)
    print("********"*10)
    text2id = processor.gen_text2id(text, vocab, padding=True, max_len=10)
    print(text2id)



