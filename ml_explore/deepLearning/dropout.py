#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：dropout.py
#   创 建 者：YuLianghua
#   创建日期：2020年07月02日
#   描    述：
#
#================================================================

def dropout(x, level):
    #level是概率值，必须在0~1之间
    if level < 0. or level >= 1:
        raise Exception('Dropout level must be in interval [0, 1[.')
    retain_prob = 1. - level
    sample=np.random.binomial(n=1, p=retain_prob, size=x.shape)
    x *=sample

    # NOTE
    x /= retain_prob

    return x
