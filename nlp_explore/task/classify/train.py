#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：train.py
#   创 建 者：YuLianghua
#   创建日期：2019年11月28日
#   描    述：
#
#================================================================
import os, sys
ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-3])
print(ROOT_PATH)
sys.path.append(ROOT_PATH)

import tensorflow as tf

from utils.data_helper import DataHelper
from utils.data_helper import train_test_split


def train(x_train, y_train, x_dev, y_dev):
    pass

def preprocess(data_helper):
    vocab = data_helper.load_vocab()
    data_list = data_helper.get_data(id_fields=['x'])
    x, y = data_list['x'], data_list['y']
    x, max_document_length = data_helper.padding(x, maxlen=50)
    all_data = zip(x, y)
    x_train, y_train, x_test, y_test = train_test_split(x, y)

if __name__ == "__main__":
    fpath = "../../data/classify/data.csv"
    vocab_path = "../../data/vocab.txt"
    data_helper = DataHelper(fpath, vocab_path, fields=['y', 'x'], startline=1)
    # x_train, y_train, x_dev, y_dev = preprocess(data_helper)
    preprocess(data_helper)




