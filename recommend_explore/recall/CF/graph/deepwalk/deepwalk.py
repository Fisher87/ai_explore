#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：deepwalk.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月07日
#   描    述：
#       reference: [1]. https://github.com/shenweichen/GraphEmbedding/blob/master/ge/models/deepwalk.py
#
#================================================================

from Skipgram import Skipgram


class DeepWalk(object):
    def __init__(self):
        self.skipgram_model = Skipgram()
        pass

    def random_walk(self):
        pass

    def skipgram(self):
        pass

    def train(self):
        pass
