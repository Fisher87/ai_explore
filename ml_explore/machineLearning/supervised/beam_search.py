#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：beam_search.py
#   创 建 者：Yulianghua
#   创建日期：2020年08月01日
#   描    述：
#
#================================================================
import numpy as np

def beam_search_decode(score, k=2):
    """beam search .
    @param score: (seq_len, num_tags)
    """
    treills = np.zeros_like([score.shape[0], k])
    # treills[0] = treils[np.argsort()[-2:]]

data = [[0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1]]
data = np.array(data)