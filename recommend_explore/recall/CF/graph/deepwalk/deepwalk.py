#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：deepwalk.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月07日
#   描    述：
#       reference: [1]. https://github.com/shenweichen/GraphEmbedding/blob/master/ge/models/deepwalk.py
#
#================================================================

import random
import itertools
from joblib import Parallel, delayed

from skipgram import Skipgram

class DeepWalk(object):
    def __init__(self, graph, walk_length, num_walks, workers=1):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks   = num_walks
        self.workers     = workers
        self.sentences = self.random_walk(verbose=1)

    def random_walk(self, verbose=0):
        def partition_num(num, workers):
            if num % workers == 0:
                return [num//workers] * workers
            else :
                return [num//workers]*workers + [num % workers]
        graph = self.graph
        nodes = list(graph.nodes())

        results = Parallel(n_jobs=self.workers, verbose=verbose)(
            delayed(self.simulate_walks)(nodes, num, self.walk_length) for num in 
            partition_num(self.num_walks, self.workers)
        )
        print(len(results))

        walks = list(itertools.chain(*results))
        return walks

    def simulate_walks(self, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deep_walk(walk_length=walk_length, start_node=v))

        return walks

    def deep_walk(self, walk_length, start_node):
        """
        DFS
        """
        walk =[start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.graph.neighbors(cur))
            if len(cur_nbrs)>0:
                walk.append(random.choice(cur_nbrs))
            else:
                break

        return walk

    def train(self, embed_size=128, w_size=5, workers=3, iter_num=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["workers"] = workers
        kwargs["window"]  = w_size
        kwargs["size"] = embed_size
        kwargs["iter"] = iter_num
        skipgram = Skipgram(**kwargs)
        self.w2v_model= skipgram

        return skipgram

    def get_embedding_all(self):
        if self.w2v_model is None:
            return {}
        self.embeddings={}
        for v in self.graph.nodes():
            self.embeddings[v] = self.w2v_model.wv[v]

        return self.embeddings

    def get_embedding_v(self, v):
        if v not in set(self.graph.nodes()):
            print("node {0} not in graph!".format(v))
            return None
        if self.embedding is None:
            return self.w2v_model.wv[v]
        else:
            return self.embeddings[v]



