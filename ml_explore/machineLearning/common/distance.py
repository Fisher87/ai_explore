#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：distance.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月20日
#   描    述：各种距离方式的python实现；
#              1. Euclidean Distance
#              2. Manhattan Distance
#              3. Chebyshev Distance
#              4. Cosine Distance
#              5. Hamming Distance
#              6. Jaccard similarity coefficient
#================================================================

import numpy as np

# 1. Euclidean Distance
def euclidean(v1, v2):
    '''
    @param v1: np.array
    '''
    return np.math.sqrt((v1-v2)*((v1-v2).T))

# 2. Manhattan Distance
def manhattan(v1, v2):
    return sum(abs(v1 - v2))

# 3. Chebyshev Distance
def chebyshev(v1, v2):
    return abs(v1 - v2).max()

# 4. Cosine Distance
def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 5. Hamming Distance
def hamming(v1, v2):
    smstr = np.nonzero(v1 - v2)
    return np.shape(smstr)[1]

# 6. Jaccard similarity coefficient
def jaccard_sim(v1, v2):
    s1 = set(v1)
    s2 = set(v2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


