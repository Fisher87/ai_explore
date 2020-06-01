#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：kmeans.py
#   创 建 者：YuLianghua
#   创建日期：2020年06月01日
#   描    述：
#
#================================================================
import sklearn
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans

from tokenizer import TokenHandler

config = defaultdict(dict)
config['tokenizer']['user_dict'] = './user_dict.txt'
config['tokenizer']['char_map'] = './char_map.txt'
tokenizer = TokenHandler(config, postag=False)

stopwords = set([s.strip() for s in open('./stopwords.txt', 'r').readlines()])

# get corpus: [['word1, word2, ...., wordn'], ..., [...]]
corpus = []

## tfidf
vectorizer = CountVectorizer()
transformer= TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

word = vectorizer.get_feature_names()
tfidf_weight = tfidf.toarray()

# PCA   
pca = sklearn.decomposition.PCA(n_components=300)
pca.fit(tfidf_weight)
new_tw = pca.transform(tfidf_weight)

## Kmeans cluster
kmeans = KMeans(n_clusters=3000)
kmeans.fit(new_tw)

print(kmeans.cluster_centers_)
for index, label in enumerate(kmeans.labels_, 1):
    print("index: {}, label: {}".format(index, label))

print("inertia: {}".format(kmeans.inertia_))
