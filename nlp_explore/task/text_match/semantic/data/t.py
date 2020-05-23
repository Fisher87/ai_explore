#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：t.py
#   创 建 者：YuLianghua
#   创建日期：2020年05月23日
#   描    述：
#
#================================================================

import pandas as pd

# def load_data(fpath):
#     df = pd.read_csv(fpath)
#     p = df['sentence1'].values
#     h = df['sentence2'].values
#     label = df['label'].values
#     lines = []
#     for i in range(len(p)):
#         line = '\t'.join([p[i], h[i], str(label[i])])
#         lines.append(line)

#     with open('./data.csv', 'w') as wf:
#         wf.write('\n'.join(lines))

# load_data('./train.csv')

with open('./data.csv', 'r') as rf:
    for line in rf:
        items = line.strip().split('\t')
        if len(items)!=3:
            print(line.strip())

