#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：lda_.py
#   创 建 者：YuLianghua
#   创建日期：2020年06月15日
#   描    述：
#
#================================================================

import numpy as np
import time
import codecs
import jieba
import re

# 预处理(分词，去停用词，为每个word赋予一个编号，文档使用word编号的列表表示)
def preprocessing():
    # 读取停止词文件
    file = codecs.open('stopwords.dic','r','utf-8')
    stopwords = [line.strip() for line in file] 
    file.close()
    
    # 读数据集
    file = codecs.open('dataset.txt','r','utf-8')
    documents = [document.strip() for document in file] 
    file.close()
    
    word2id = {}
    id2word = {}
    docs = []
    currentDocument = []
    currentWordId = 0
    
    for document in documents:
        # 分词
        segList = jieba.cut(document)
        for word in segList: 
            word = word.lower().strip()
            # 单词长度大于1并且不包含数字并且不是停止词
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
                if word in word2id:
                    currentDocument.append(word2id[word])
                else:
                    currentDocument.append(currentWordId)
                    word2id[word] = currentWordId
                    id2word[currentWordId] = word
                    currentWordId += 1
        docs.append(currentDocument);
        currentDocument = []
    return docs, word2id, id2word
    
# 初始化，按照每个topic概率都相等的multinomial分布采样，等价于取随机数，并更新采样出的topic的相关计数
def randomInitialize():
    for d, doc in enumerate(docs):
        zCurrentDoc = []
        for w in doc:
            # 主题分布, ndz: 每篇文档中每个topic产生的单词数; nzw: 每个topic产生单词w的计数; nz: 每个topic产生全部单词的计数;
            pz = np.divide(np.multiply(ndz[d, :], nzw[:, w]), nz)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            zCurrentDoc.append(z)
            ndz[d, z] += 1
            nzw[z, w] += 1
            nz[z] += 1
        Z.append(zCurrentDoc)

# gibbs采样
def gibbsSampling():
    # 为每个文档中的每个单词重新采样topic
    for d, doc in enumerate(docs):
        for index, w in enumerate(doc):
            z = Z[d][index]
            # 将当前文档当前单词原topic相关计数减去1
            ndz[d, z] -= 1
            nzw[z, w] -= 1
            nz[z] -= 1
            # 重新计算当前文档当前单词属于每个topic的概率
            pz = np.divide(np.multiply(ndz[d, :], nzw[:, w]), nz)
            # 按照计算出的分布进行采样
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            Z[d][index] = z 
            # 将当前文档当前单词新采样的topic相关计数加上1
            ndz[d, z] += 1
            nzw[z, w] += 1
            nz[z] += 1

def perplexity():
    nd = np.sum(ndz, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(docs):
        for w in doc:
            ll = ll + np.log(((nzw[:, w] / nz) * (ndz[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll/(-n))



# dir分布超参, 生成主题分布模型(多项分布)参数
alpha = 5
# dir分布超参, 生成词分布模型(多项分布)参数
beta = 0.1    
iterationNum = 50
Z = []
K = 10  # 主题数
docs, word2id, id2word = preprocessing()
N = len(docs)   # 文档总数
M = len(word2id)   # 词总数

# 每篇文档d对于的topic分布
ndz = np.zeros([N, K]) + alpha
# 每个主题topic对应的词分布
nzw = np.zeros([K, M]) + beta
# 每个主题产生的全部单词个数
nz = np.zeros([K]) + M * beta
randomInitialize()
for i in range(0, iterationNum):
    gibbsSampling()
    print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", perplexity())
 
topicwords = []
maxTopicWordsNum = 10
for z in range(0, K):
    ids = nzw[z, :].argsort()
    topicword = []
    for j in ids:
        topicword.insert(0, id2word[j])
    topicwords.append(topicword[0 : min(10, len(topicword))])
