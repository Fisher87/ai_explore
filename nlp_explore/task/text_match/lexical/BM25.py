#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：BM25.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月03日
#   描    述：
#       refer:https://www.jianshu.com/p/1e498888f505
#
#================================================================

import math
import jieba
import utils
from collections import defaultdict


class BM25(object):
    def __init__(self, docs):
        """
        self.f: 每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df: 每个词及出现了该词的文档数量;
        self.idf: 每个词对应的idf值，q_i 对应的权重W_i
        """
        self.D = len(docs)
        self.avgdl = sum([len(d) for d in docs]) / float(self.D)
        self.docs = docs
        self.f = [] 
        self.df = defaultdict(int)
        self.idf = {}
        self.k1 = 2
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = defaultdict(int)
            for word in doc:
                tmp[word] += 1
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] += 1

        for k, v in self.df.items():
            # idf = log( (N-n+0.5) / (n+0.5) )   
            self.idf[k] = math.log(self.D -v + 0.5) - math.log(v+0.5)

    def sim(self, doc, index):
        """
        score = 0
        for i in n:    # n表示query的语素q_i个数
            score += idf(q_i) * ( f_i*(k1+1) / (f_i+k1*(1-b+b*dl/avgdl) ) )
        end
        """
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word]*self.f[index][word]*(self.k1+1) / 
                      (self.f[index][word] + self.k1*(1-self.b+self.b*d/self.avgdl)))

        return score

    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores

if __name__ == "__main__":
    text = """
    自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
    它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
    自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
    因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
    所以它与语言学的研究有着密切的联系，但又有重要的区别。
    自然语言处理并不是一般地研究自然语言，
    而在于研制能有效地实现自然语言通信的计算机系统，
    特别是其中的软件系统。因而它是计算机科学的一部分。
    """
    sents = utils.get_sentences(text)
    doc = []
    for sent in sents:
        words = list(jieba.cut(sent))
        words = utils.filter_stop(words)
        doc.append(words)
    print(doc)
    s = BM25(doc)
    print(s.f)
    print(s.idf)
    print(s.simall(['自然语言', '计算机科学', '领域', '人工智能', '领域']))
