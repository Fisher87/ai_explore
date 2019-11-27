#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：test_preprocess.py
#   创 建 者：YuLianghua
#   创建日期：2019年11月27日
#   描    述：
#
#================================================================
import sys, os
ROOT_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1])
sys.path.append(ROOT_PATH)

import unittest
from preprocess.data_process import DataProcessor

data_process = DataProcessor()

class TestDataProcessor(unittest.TestCase):
    def test_gen_vocab(self):
        content = "这些方法是在一个固定的图上直接学习每个节点embedding，但是大多情况图是会演化的，当网络结构改变以及新节点的出现，直推式学习需要重新训练（复杂度高且可能会导致embedding会偏移），很难落地在需要快速生成未知节点embedding的机器学习系统上。"
        vocab = data_process.gen_vocab(content)
        vocab = data_process.gen_vocab(content, max_length=30, min_freq=2)
        print(vocab)

if __name__ == "__main__":
    unittest.main()


