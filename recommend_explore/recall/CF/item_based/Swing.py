#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：Swing.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月05日
#   描    述：
#
#================================================================

import numpy as np

# sim(i, j) = \sum_{u}^{U_i} \sum_{v}^{V_j} (1 / (\alpha+| I_u & I_v |) )   `&`表示取交集
#    U_i: purchase item i all users;
#    U_j: purchase item j all users;
#    I_u: all items user u purchased;
#    I_v: all items user v purchased;
def swing(u2iM, i, j, alpha=7):
    """
    @param u2iM: user 2 item 的行为matrix [m, n];
    @param i: item i;
    @param j: item j;
    @param alpha: 
    """
    items_len = u2iM.shape[-1]
    # get item i all users
    I_users_items = u2iM[u2iM[:, i]==1]
    # get item j all users
    J_users_items = u2iM[u2iM[:, j]==1]

    sim_i_j = 0.0
    for u_items in I_users_items:
        for v_items in J_users_items:
            intersection = 0
            for i in range(items_len):
                if u_items[i] == v_items[i] == 1:
                    intersection += 1
            sim_i_j += 1 / (alpha + intersection)

    return sim_i_j

def SwingRecall(u2items):
    """
    这个实现方式跟算法定义的有点区别！
        reference: https://www.jianshu.com/p/a5d46cdc2b4e
    """
    u2Swing = defaultdict(lambda:dict())
    for u in u2items:
        wu = pow(len(u2items[u])+5,-0.35)
        for v in u2items:
            if v == u:
                continue
            wv = wu*pow(len(u2items[v])+5,-0.35)
            inter_items = set(u2items[u]).intersection(set(u2items[v]))
            for i in inter_items:
                for j in inter_items:
                    if j==i:
                        continue
                    if j not in u2Swing[i]:
                        u2Swing[i][j] = 0
                    u2Swing[i][j] += wv/(1+len(inter_items))
    return u2Swing


if __name__ == "__main__":
    M = np.array([[0, 1, 1, 0, 1], [1, 0, 1, 0, 1], [0, 0, 1, 1, 1]])
    sim = swing(M, 2, 4)
    print(sim)
