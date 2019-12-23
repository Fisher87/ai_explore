#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：Alias_method.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月10日
#   描    述：
#        reference: https://zhuanlan.zhihu.com/p/54867139
#
#================================================================
"""
Alias method 样本采样，时间复杂度为O(1).

"""

import numpy as np

def gen_prob_dist(N):
    prob = np.random.randint(0, 100, N)
    return prob/np.sum(prob)

def create_alias_table(area_ratio):
    """
    1. 确保每个方格的area值为1；
    2. 确保每个方格区域最多只有两个类别;

    """
    l = len(area_ratio)
    accept, alias = [0]*l, [0]*l
    small, large = [], []

    for i, p in enumerate(area_ratio):
        if p<1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio[small_idx]
        alias[small_idx] = large_idx
        area_ratio[large_idx] = area_ratio[large_idx] - (1-area_ratio[small_idx])
        if area_ratio[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias

def alias_sample(accept, alias):
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else :
        return alias[i]

def main(N=100, k=10000):
    # generate prob distribute.
    prob = gen_prob_dist(N) 
    # print(prob)

    # p_i * N, generate area ratio.
    area = prob * N
    # print(area)

    # create alias table
    accept, alias = create_alias_table(area)
    # print(len(accept))
    # print(len(alias))

    ans = np.zeros(N)
    for _ in range(k):
        i = alias_sample(accept, alias)
        ans[i] += 1
    print(ans/np.sum(ans))
    print(prob)

if __name__=="__main__":
    main()


