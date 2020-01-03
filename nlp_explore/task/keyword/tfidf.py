#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：tfidf.py
#   创 建 者：YuLianghua
#   创建日期：2020年01月03日
#   描    述：
#
#================================================================
# coding:utf-8
import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus=["我 来到 北京 清华大学",
    "他 来到 了 网易 杭研 大厦",
    "小明 硕士 毕业 与 中国 科学院",
    "我 爱 北京 天安门"]
vectorizer=CountVectorizer()         #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer=TfidfTransformer()       #该类会统计每个词语的tf-idf权值
X=vectorizer.fit_transform(corpus)   #将文本转为词频矩阵
tfidf=transformer.fit_transform(X)   #计算tf-idf，
word=vectorizer.get_feature_names()  #获取词袋模型中的所有词语
weight=tfidf.toarray()               #将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
for i in range(len(weight)):         #打印每类文本的tf-idf词语权重
    print("-------这里输出第",i,u"类文本的词语tf-idf权重------" )
    #for j in range(len(word)):
    print(list(zip(word,weight[i])))


