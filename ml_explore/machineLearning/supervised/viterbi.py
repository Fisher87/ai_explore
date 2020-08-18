#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：viterbi.py
#   创 建 者：YuLianghua
#   创建日期：2020年08月01日
#   描    述：维特比解码算法
#
#================================================================
import numpy as np

def viterbi_decode(score, transition_params):
    """
    This should only be used at test time.
    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag
          indices.
      viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score
