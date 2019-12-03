#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：LSI.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月03日
#   描    述：
#      refer: https://blog.csdn.net/qq_16633405/article/details/80577851
#
#================================================================

import jieba
from gensim import corpora, models
from gensim.similarities import Similarity

jieba.load_userdict("userdict.txt")
stopwords = set(open('stopwords.txt',encoding='utf8').read().strip('\n').split('\n'))   #读入停用词
raw_documents = [
    '0无偿居间介绍买卖毒品的行为应如何定性',
    '1吸毒男动态持有大量毒品的行为该如何认定',
    '2如何区分是非法种植毒品原植物罪还是非法制造毒品罪',
    '3为毒贩贩卖毒品提供帮助构成贩卖毒品罪',
    '4将自己吸食的毒品原价转让给朋友吸食的行为该如何认定',
    '5为获报酬帮人购买毒品的行为该如何认定',
    '6毒贩出狱后再次够买毒品途中被抓的行为认定',
    '7虚夸毒品功效劝人吸食毒品的行为该如何认定',
    '8妻子下落不明丈夫又与他人登记结婚是否为无效婚姻',
    '9一方未签字办理的结婚登记是否有效',
    '10夫妻双方1990年按农村习俗举办婚礼没有结婚证 一方可否起诉离婚',
    '11结婚前对方父母出资购买的住房写我们二人的名字有效吗',
    '12身份证被别人冒用无法登记结婚怎么办？',
    '13同居后又与他人登记结婚是否构成重婚罪',
    '14未办登记只举办结婚仪式可起诉离婚吗',
    '15同居多年未办理结婚登记，是否可以向法院起诉要求离婚'
]
corpora_documents = []
for item_text in raw_documents:
    item_str = jieba.lcut(item_text)
    print(item_str)
    corpora_documents.append(item_str)
# 生成字典和向量语料
dictionary = corpora.Dictionary(corpora_documents)
print("dictionary"+str(dictionary))
# 通过下面一句得到语料中每一篇文档对应的稀疏向量（这里是bow向量）
corpus = [dictionary.doc2bow(text) for text in corpora_documents]
# 向量的每一个元素代表了一个word在这篇文档中出现的次数
print("corpus:"+str(corpus))
# 测试数据
test_data_1 = '你好，我想问一下我想离婚他不想离，孩子他说不要，是六个月就自动生效离婚'
test_cut_raw_1 = jieba.lcut(test_data_1)

print(test_cut_raw_1)
#转化成tf-idf向量
# corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
tfidf_model=models.TfidfModel(corpus)
corpus_tfidf = [tfidf_model[doc] for doc in corpus]
print(corpus_tfidf)
#转化成lsi向量
lsi= models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=50)
corpus_lsi = [lsi[doc] for doc in corpus]
print("corpus_lsi："+str(corpus_lsi))
similarity_lsi=Similarity('Similarity-Lsi-index', corpus_lsi, num_features=400,num_best=5)
# 2.转换成bow向量 # [(51, 1), (59, 1)]，即在字典的52和60的地方出现重复的字段，这个值可能会变化
test_corpus_3 = dictionary.doc2bow(test_cut_raw_1)  
print(test_corpus_3)
# 3.计算tfidf值  # 根据之前训练生成的model，生成query的IFIDF值，然后进行相似度计算
test_corpus_tfidf_3 = tfidf_model[test_corpus_3]  
print(test_corpus_tfidf_3) # [(51, 0.7071067811865475), (59, 0.7071067811865475)]
# 4.计算lsi值
test_corpus_lsi_3 = lsi[test_corpus_tfidf_3]  
print(test_corpus_lsi_3)
# lsi.add_documents(test_corpus_lsi_3) #更新LSI的值
print('——————————————lsi———————————————')
# 返回最相似的样本材料,(index_of_document, similarity) tuples
print(similarity_lsi[test_corpus_lsi_3])


