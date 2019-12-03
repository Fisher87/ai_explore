#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：utils.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月03日
#   描    述：
#
#================================================================
import os
import re
import codecs


stop_path = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-4]), 'data/stopwords.txt')

stop = set()
fr = codecs.open(stop_path, 'r', 'utf-8')
for word in fr:
    stop.add(word.strip())
fr.close()
re_zh = re.compile('([\u4E00-\u9FA5]+)')


def filter_stop(words):
    return list(filter(lambda x: x not in stop, words))


def get_sentences(doc):
    line_break = re.compile('[\r\n]')
    delimiter = re.compile('[，。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
    return sentences



