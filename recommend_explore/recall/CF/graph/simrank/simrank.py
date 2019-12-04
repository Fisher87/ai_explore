#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：simrank.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月03日
#   描    述：
#      Reference: 
#            [1]. https://www.cnblogs.com/pinard/p/6362647.html
#
#================================================================

from itertools import product
import networkx as nx
import numpy as np

def _is_close(d1, d2, atolerance=0, rtolerance=0):
    """Determines whether two adjacency matrices are within
    a provided tolerance.

    Parameters
    ----------
    d1 : dict
        Adjacency dictionary

    d2 : dict
        Adjacency dictionary

    atolerance : float
        Some scalar tolerance value to determine closeness

    rtolerance : float
        A scalar tolerance value that will be some proportion
        of ``d2``'s value

    Returns
    -------
    closeness : bool
        If all of the nodes within ``d1`` and ``d2`` are within
        a predefined tolerance, they are considered "close" and
        this method will return True. Otherwise, this method will
        return False.

    """
    # Pre-condition: d1 and d2 have the same keys at each level if they
    # are dictionaries.
    if not isinstance(d1, dict) and not isinstance(d2, dict):
        return abs(d1 - d2) <= atolerance + rtolerance * abs(d2)
    return all(all(_is_close(d1[u][v], d2[u][v]) for v in d1[u]) for u in d1)


def simrank(G, 
            source=None, 
            target=None, 
            importance_factor=0.9,
            max_iterations=100,
            tolerance=1e-4):
    """
    The pseudo-code definition from the paper is::
        def simrank(G, u, v):
            in_neighbors_u = G.predecessors(u)
            in_neighbors_v = G.predecessors(v)
            scale = C / (len(in_neighbors_u) * len(in_neighbors_v))
            return scale * sum(simrank(G, w, x)
                               for w, x in product(in_neighbors_u,
                                                   in_neighbors_v))
    """
    prevsim = None

    # build up our similarity adjacency dictionary output
    newsim = {u: {v: 1 if u == v else 0 for v in G} for u in G}

    # These functions compute the update to the similarity value of the nodes
    # `u` and `v` with respect to the previous similarity values.
    def avg_sim(s):
        """
            len(s) is euqal (I(a) * I(b))
        """
        return sum(newsim[w][x] for (w, x) in s) / len(s) if s else 0.0

    def sim(u, v):
        return importance_factor * avg_sim(list(product(G[u], G[v])))

    for _ in range(max_iterations):
        if prevsim and _is_close(prevsim, newsim, tolerance):
            break
        prevsim = newsim
        newsim = {
            u: {v: sim(u, v) if u is not v else 1 for v in newsim[u]} for u in newsim
        }

    if source is not None and target is not None:
        return newsim[source][target]
    if source is not None:
        return newsim[source]
    return newsim

def main():
    #init graph
    G = nx.Graph()
    G.add_nodes_from([i for i in range(1, 10)])
    edges = [(1,5), (1,6),(1,7),(2,5),(2,6),(2,7),(2,8),(3,7),(3,9),(4,8),(4,9)]
    G.add_edges_from(edges)
    print(G[1])
    print(list(product(G[1], G[2])))
    # newsim = simrank(G, importance_factor=0.7)
    # print(newsim)


if __name__=="__main__":
    main()
