#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：sim_rank.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月06日
#   描    述：
#         simrank 算法简单实现
#         reference: https://github.com/littleq0903/simrank/blob/master/simrank.py
#
#================================================================
import numpy
from numpy import matrix

with open('sample.txt', 'r') as log_fp:
    logs = [ log.strip() for log in log_fp.readlines() ]

logs_tuple = [ tuple(log.split(",")) for log in logs ]

queries = list(set([ log[0] for log in logs_tuple ]))
ads = list(set([ log[1] for log in logs_tuple ]))

# Graph means the relations number
graph = numpy.matrix(numpy.zeros([len(queries), len(ads)]))

for log in logs_tuple:
    query = log[0]
    ad = log[1]
    q_i = queries.index(query)
    a_j = ads.index(ad)
    graph[q_i, a_j] += 1

print graph

query_sim = matrix(numpy.identity(len(queries)))
ad_sim = matrix(numpy.identity(len(ads)))

def get_ads_num(query):
    q_i = queries.index(query)
    return graph[q_i]

def get_queries_num(ad):
    a_j = ads.index(ad)
    return graph.transpose()[a_j]

def get_ads(query):
    series = get_ads_num(query).tolist()[0]
    return [ ads[x] for x in range(len(series)) if series[x] > 0 ]

def get_queries(ad):
    series = get_queries_num(ad).tolist()[0]
    return [ queries[x] for x in range(len(series)) if series[x] > 0 ]


def query_simrank(q1, q2, C):
    """
    in this, graph[q_i] -> connected ads
    """
    """
    print "q1.ads"
    print get_ads_num(q1).tolist()
    print "q2.ads"
    print get_ads_num(q2).tolist()
    """
    if q1 == q2 : return 1
    prefix = C / (get_ads_num(q1).sum() * get_ads_num(q2).sum())
    postfix = 0
    for ad_i in get_ads(q1):
        for ad_j in get_ads(q2):
            i = ads.index(ad_i)
            j = ads.index(ad_j)
            postfix += ad_sim[i, j]
    return prefix * postfix


def ad_simrank(a1, a2, C):
    """
    in this, graph need to be transposed to make ad to be the index
    """
    """
    print "a1.queries"
    print get_queries_num(a1)
    print "a2.queries"
    print get_queries_num(a2)
    """
    if a1 == a2 : return 1
    prefix = C / (get_queries_num(a1).sum() * get_queries_num(a2).sum())
    postfix = 0
    for query_i in get_queries(a1):
        for query_j in get_queries(a2):
            i = queries.index(query_i)
            j = queries.index(query_j)
            postfix += query_sim[i,j]
    return prefix * postfix


def simrank(C=0.8, times=1):
    global query_sim, ad_sim

    for run in range(times):
        # queries simrank
        new_query_sim = matrix(numpy.identity(len(queries)))
        for qi in queries:
            for qj in queries:
                i = queries.index(qi)
                j = queries.index(qj)
                new_query_sim[i,j] = query_simrank(qi, qj, C)

        # ads simrank
        new_ad_sim = matrix(numpy.identity(len(ads)))
        for ai in ads:
            for aj in ads:
                i = ads.index(ai)
                j = ads.index(aj)
                new_ad_sim[i,j] = ad_simrank(ai, aj, C)

        query_sim = new_query_sim
        ad_sim = new_ad_sim


if __name__ == '__main__':
    print queries
    print ads
    simrank()
    print query_sim
    print ad_sim


