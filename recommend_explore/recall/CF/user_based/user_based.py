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

import math
import time
from typing import List


def _user_similarity(train_data: list, n_user: int, n_item: int) -> List[List[float]]:
    print('开始计算每两个用户之间的相似度。')
    start_time = time.time()

    train_item_users = [[] for _ in range(n_item)]  # train_item_users[i]是对物品i有过正反馈的所有用户列表
    N = [0 for _ in range(n_user)]  # N[u]是用户u有过正反馈的所有物品的数量
    for user_id, item_id, _ in train_data:
        train_item_users[item_id].append(user_id)
        N[user_id] += 1

    # 统计每两个用户之间的共同正反馈物品数量和每个用户有过正反馈物品的总量
    W = [[0 for _ in range(n_user)] for _ in range(n_user)]  # W[u][v]是用户u和v的共同有正反馈物品的数量（v>u）
    for users in train_item_users:
        for u in users:
            for v in users:
                if v > u:
                    W[u][v] += 1
    # 计算相似度
    for i in range(n_user - 1):
        for j in range(i + 1, n_user):
            if W[i][j] != 0:
                W[i][j] /= math.sqrt(N[i] * N[j])
                W[j][i] = W[i][j]

    print('（耗时', time.time() - start_time, '秒）', sep='')
    return W


def _user_item_score(train_data: list, n_user: int, n_item: int, W: List[List[float]], N=80) -> List[List[float]]:
    print('开始计算用户物品评分矩阵，', 'N=', N, sep='')
    start_time = time.time()

    # 得到训练集中每个用户所有有过正反馈物品集合
    train_user_items = [set() for _ in range(n_user)]
    for user_id, item_id, _ in train_data:
        train_user_items[user_id].add(item_id)

    scores = [[0 for _ in range(n_item)] for _ in range(n_user)]  # scores[u][i]是用户u对物品i的评分
    for user_id in range(n_user):
        Wu = dict()
        for v in range(n_user):
            if W[user_id][v] != 0:
                Wu[v] = W[user_id][v]

        # 计算出用户user_id对每个物品感兴趣程度
        for similar_user_id, similarity_factor in sorted(Wu.items(), key=lambda x: x[1], reverse=True)[:N]:
            for item_id in train_user_items[similar_user_id] - train_user_items[user_id]:
                scores[user_id][item_id] += similarity_factor

    print('（耗时', time.time() - start_time, '秒）', sep='')
    return scores


if __name__ == '__main__':
    n_user, n_item, train_data, test_data, topk_data = data_process()

    W = _user_similarity(train_data, n_user, n_item)
    scores = _user_item_score(train_data, n_user, n_item, W, N=80)

    ks = [10, 36, 100]
    score_fn = lambda ui: [scores[u][i] for u, i in zip(ui['user_id'], ui['item_id'])]
