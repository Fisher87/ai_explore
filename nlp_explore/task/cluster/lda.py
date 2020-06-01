#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：lda.py
#   创 建 者：YuLianghua
#   创建日期：2020年06月01日
#   描    述：text cluster by lda topic extract;
#
#================================================================

from collections import defaultdict
from gensim import corpora
from gensim import models

from tokenizer import TokenHandler

config = defaultdict(dict)
config['tokenizer']['user_dict'] = './user_dict.txt'
config['tokenizer']['char_map'] = './char_map.txt'
tokenizer = TokenHandler(config, postag=False)

stopwords = set([s.strip() for s in open('./stopwords.txt', 'r').readlines()])

# get corpus: [['word1', 'word2', ...., 'wordn'], ..., [...]]
corpus = []
dictionary = corpora.Dictionary(corpus)
n_corpus = [dictionary.doc2bow(words) for words in corpus]
# lda model
num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = False
lda = models.ldamodel.LdaModel(corpus=n_corpus, 
                               id2word=dictionary, 
                               chunksize=chunksize,
                               alpha='auto',
                               eta='auto',
                               iterations=iterations,
                               num_topics=num_topics
                               passes=passes,
                               eval_every=eval_every
                              )

# print topK words in each topic
for topic_words in lda.print_topics(num_words=5):
    print(topic_words)

# get doc topic 
for e, values in enumerate(lda.inference(n_corpus)[0]):
    values_sorted = sorted(values, key=lambda x:x[-1], reverse=True)
    topic_id, topic_val = values_sorted[0]
