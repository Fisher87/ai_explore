#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：node2vec.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月18日
#   描    述：
#
#================================================================


class Node2Vec(object):
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):
        self.graph = graph
