#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：data_process.py
#   创 建 者：YuLianghua
#   创建日期：2020年01月03日
#   描    述：
#
#================================================================
import time
import os
from typing import List, Callable, Tuple
from dataclasses import dataclass

# 记下自己的路径，确保其它py文件调用时读数据路径正确
root_path = os.path.dirname(__file__)

@dataclass
class TopkData:
    test_user_item_set: dict  # 在测试集上每个用户可以参与推荐的物品集合
    test_user_positive_item_set: dict  # 在测试集上每个用户有行为的物品集合

def _read_ml(relative_path: str, separator: str) -> List[Tuple[int, int, int, int]]:
    data = []
    with open(os.path.join(root_path, relative_path), 'r') as f:
        for line in f.readlines():
            values = line.strip().split(separator)
            user_id, movie_id, rating, timestamp = int(values[0]), int(values[1]), int(values[2]), int(values[3])
            data.append((user_id, movie_id, rating, timestamp))
    return data


def _read_ml100k() -> List[Tuple[int, int, int, int]]:
    return _read_ml('ml-100k/u.data', '\t')


def _read_ml1m() -> List[Tuple[int, int, int, int]]:
    return _read_ml('ml-1m/ratings.dat', '::')


def _read_lastfm() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(root_path, 'lastfm-2k/user_artists.dat'), 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split('\t')
            user_id, artist_id, weight = int(values[0]), int(values[1]), int(values[2])
            data.append((user_id, artist_id, weight))
    return data


def _load_data(read_data_fn: Callable[[], List[tuple]], expect_length: int, expect_user: int, expect_item: int,
               data_name: str, user_name='用户', item_name='物品') -> List[tuple]:
    print('开始读数据', data_name, '。共', expect_length, '条数据，有',
          expect_user, '个', user_name, '，', expect_item, '个', item_name, '。', sep='')
    start_time = time.time()
    data = read_data_fn()
    n_user, n_item = len(set(d[0] for d in data)), len(set(d[1] for d in data))
    assert len(data) == expect_length and n_user == expect_user and n_item == expect_item
    print('（耗时', time.time() - start_time, '秒）', sep='')
    return data


def ml100k() -> List[Tuple[int, int, int, int]]:
    return _load_data(_read_ml100k, 100000, 943, 1682, 'ml100k', item_name='电影')


def ml1m() -> List[Tuple[int, int, int, int]]:
    return _load_data(_read_ml1m, 1000209, 6040, 3706, 'ml1m', item_name='电影')


def lastfm() -> List[Tuple[int, int, int]]:
    return _load_data(_read_lastfm, 92834, 1892, 17632, 'lastfm', item_name='艺术家')

#########################################
import time
import random
import numpy as np
from typing import Tuple, Dict, List, Callable
from collections import defaultdict
from Recommender_System.utility.evaluation import TopkData


def negative_sample(data: List[tuple], ratio=1, threshold=0, method='random') -> List[Tuple[int, int, int]]:
    """
    采集负样本

    :param data: 原数据，第一列是用户id，第二列是物品id，第三列是权重
    :param ratio: 负正样本比例
    :param threshold: 权重阈值，权重大于或者等于此值为正样例，小于此值既不是正样例也不是负样例
    :param method: 采集方式，random是均匀随机采集，popular是按流行度随机采集
    :return: 带上负样本的数据集
    """
    print('开始采集负样本，负正样本比例为', ratio, '，权重阈值为', threshold, '，方法为', method, '。', sep='')
    start_time = time.time()

    # 负样本采集权重
    item_set = {d[1] for d in data}
    if method == 'random':
        negative_sample_weight = {item_id: 1 for item_id in item_set}
    elif method == 'popular':
        negative_sample_weight = {item_id: 0 for item_id in item_set}
        for d in data:
            negative_sample_weight[d[1]] += 1
    else:
        raise ValueError("参数method必须是'random'或'popular'")

    # 得到每个用户正样本集合
    user_positive_set = defaultdict(set)
    user_unpositive_set = defaultdict(set)
    for d in data:
        user_id, item_id, weight = d[0], d[1], d[2]
        if weight >= threshold:
            user_positive_set[user_id].add(item_id)
        else:
            user_unpositive_set[user_id].add(item_id)

    # 为每个用户采集负样例
    new_data = []
    for user_id, positive_set in user_positive_set.items():
        for positive_item_id in positive_set:
            new_data.append((user_id, positive_item_id, 1))  # 将正样例加入数据集

        valid_negative_list = list(item_set - positive_set - user_unpositive_set[user_id])  # 可以取负样例的物品id列表
        n_negative_sample = min(int(len(positive_set) * ratio), len(valid_negative_list))  # 采集负样例数量
        if n_negative_sample <= 0:
            continue

        sum_weight = sum([negative_sample_weight[item_id] for item_id in valid_negative_list])
        weights = [negative_sample_weight[item_id] / sum_weight for item_id in valid_negative_list]  # 负样本采集权重

        # 采集n_negative_sample个负样例
        for negative_item_id in np.random.choice(valid_negative_list, n_negative_sample, False, weights):
            new_data.append((user_id, negative_item_id, 0))  # 将负样例加入数据集

    print('（耗时', time.time() - start_time, '秒）', sep='')
    return new_data


def neaten_id(data: List[Tuple[int, int, int]]) -> Tuple[List[Tuple[int, int, int]], int, int, Dict[int, int], Dict[int, int]]:  # data前两列被视为用户id和物品id
    """
    对数据的用户id和物品id进行规整化，使其id变为从0开始到数量减1

    :param data: 原数据，第一列是用户id，第二列是物品id，第三列是标签
    :return: 新数据，用户数量，物品数量，用户id旧到新映射，物品id旧到新映射
    """
    print('开始进行id规整化。')
    start_time = time.time()

    new_data = []
    n_user, n_item = 0, 0
    user_id_old2new, item_id_old2new = {}, {}
    for user_id_old, item_id_old, label in data:
        if user_id_old not in user_id_old2new:
            user_id_old2new[user_id_old] = n_user
            n_user += 1
        if item_id_old not in item_id_old2new:
            item_id_old2new[item_id_old] = n_item
            n_item += 1
        new_data.append((user_id_old2new[user_id_old], item_id_old2new[item_id_old], label))

    print('（耗时', time.time() - start_time, '秒）', sep='')
    return new_data, n_user, n_item, user_id_old2new, item_id_old2new


def split(data: List[tuple], test_ratio=0.2, shuffle=True) -> Tuple[List[tuple], List[tuple]]:
    """
    将数据切分为训练集数据和测试集数据

    :param data: 原数据
    :param test_ratio: 测试集数据占比，这个值在0和1之间
    :param shuffle: 是否对原数据随机排序
    :return: 训练集数据和测试集数据
    """
    print('开始数据切分，test_ratio=', test_ratio, ', shuffle=', shuffle, sep='')
    start_time = time.time()

    if shuffle:
        random.shuffle(data)
    n_test = int(len(data) * test_ratio)
    test_data, train_data = data[:n_test], data[n_test:]

    print('（耗时', time.time() - start_time, '秒）', sep='')
    return train_data, test_data


def prepare_topk(train_data: List[Tuple[int, int, int]], test_data: List[Tuple[int, int, int]],
                 n_user: int, n_item: int, n_sample_user=None) -> TopkData:
    """
    准备用于topk评估的数据

    :param train_data: 训练集数据，有三列，分别是user_id, item_id, label
    :param test_data: 测试集数据，有三列，分别是user_id, item_id, label
    :param n_user: 用户数量
    :param n_item: 物品数量
    :param n_sample_user: 用户取样数量，为None则表示采样所有用户
    :return: 用于topk评估的数据，类型为TopkData，其包括在测试集里每个用户的（可推荐物品集合）与（有行为物品集合）
    """
    if n_sample_user is None or n_sample_user > n_user:
        n_sample_user = n_user
    print('开始准备topk评估数据，n_sample_user=', n_sample_user, sep='')
    start_time = time.time()

    user_set = np.random.choice(range(n_user), n_sample_user, False)

    def get_user_item_set(data: List[Tuple[int, int, int]], only_positive=False):
        user_item_set = {user_id: set() for user_id in user_set}
        for user_id, item_id, label in data:
            if user_id in user_set and (not only_positive or label == 1):
                user_item_set[user_id].add(item_id)
        return user_item_set

    test_user_item_set = {user_id: set(range(n_item)) - item_set
                          for user_id, item_set in get_user_item_set(train_data).items()}
    test_user_positive_item_set = get_user_item_set(test_data, only_positive=True)

    print('（耗时', time.time() - start_time, '秒）', sep='')
    return TopkData(test_user_item_set, test_user_positive_item_set)


def pack(data_loader_fn: Callable[[], List[tuple]],
         negative_sample_ratio=1, negative_sample_threshold=0, negative_sample_method='random',
         split_test_ratio=0.2, shuffle_before_split=True,
         topk_sample_user=300) -> Tuple[int, int, List[Tuple[int, int, int]], List[Tuple[int, int, int]], TopkData]:
    """
    读数据，负采样，训练集测试集切分，准备TopK评估数据

    :param data_loader_fn: data_loader里面的读数据函数
    :param negative_sample_ratio: 负正样本比例，为0代表不采样
    :param negative_sample_threshold: 负采样的权重阈值，权重大于或者等于此值为正样例，小于此值既不是正样例也不是负样例
    :param negative_sample_method: 负采样方法，值为'random'或'popular'
    :param split_test_ratio: 切分时测试集占比，这个值在0和1之间
    :param shuffle_before_split: 切分前是否对数据集随机顺序
    :param topk_sample_user: 用来计算TopK指标时用户采样数量，为None则表示采样所有用户
    :return: 用户数量，物品数量，训练集，测试集，用于TopK评估数据
    """
    data = data_loader_fn()
    if negative_sample_ratio > 0:
        data = negative_sample(data, ratio=negative_sample_ratio, threshold=negative_sample_threshold, method=negative_sample_method)
    else:
        data = [(d[0], d[1], 1) for d in data]  # 变成隐反馈数据
    data, n_user, n_item, _, _ = neaten_id(data)
    train_data, test_data = split(data, test_ratio=split_test_ratio, shuffle=shuffle_before_split)
    topk_data = prepare_topk(train_data, test_data, n_user, n_item, n_sample_user=topk_sample_user)
    return n_user, n_item, train_data, test_data, topk_data


def pack_kg(kg_loader_fn: Callable[[], Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], int, int, int, int]],
            split_test_ratio=0.2, shuffle_before_split=True, topk_sample_user=300) -> Tuple[int, int, int, int,
            List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int]], TopkData]:
    """
    联合读数据和知识图谱，训练集测试集切分，准备TopK评估数据

    :param kg_loader_fn: kg_loader里面的读数据函数
    :param split_test_ratio: 切分时测试集占比，这个值在0和1之间
    :param shuffle_before_split: 切分前是否对数据集随机顺序
    :param topk_sample_user: 用来计算TopK指标时用户采样数量，为None则表示采样所有用户
    :return: 用户数量，物品数量，实体数量，关系数量，训练集，测试集，知识图谱，用于TopK评估数据
    """
    data, kg, n_user, n_item, n_entity, n_relation = kg_loader_fn()
    train_data, test_data = split(data, test_ratio=split_test_ratio, shuffle=shuffle_before_split)
    topk_data = prepare_topk(train_data, test_data, n_user, n_item, n_sample_user=topk_sample_user)
    return n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data

