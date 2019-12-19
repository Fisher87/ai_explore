#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：user_based.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月16日
#   描    述：
#
#================================================================
import numpy as np

def generate_data(users=100, items=1000):
    """
    生成user-item评分矩阵
    """
    user_item_table = np.zeros([users, items], dtype=np.int32)
    for i in range(users):
        for j in range(items):
            p = np.random.random()
            if p <= 0.9:
                user_item_table[i][j] = 0
            else:
                user_item_table[i][j] = np.random.randint(1,6)

    return user_item_table

if __name__ == "__main__":
    user_item_table = generate_data()
    print(user_item_table[0])

