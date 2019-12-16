#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：main.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月16日
#   描    述：
#       reference: [1]. https://github.com/shenweichen/GraphEmbedding
#
#================================================================

import networkx as nx
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import plt

from deepwalk import DeepWalk
from classify import read_node_label, Classifier 

def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    G = nx.read_edgelist('../data/Wiki_edgelist.txt',
                        create_using=nx.DiGraph(), nodetype=None, data=[("weight", int)])
    print(len(G.nodes()))
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(w_size=5, iter_num=3)
    embeddings = model.get_embedding_all()
   
    evaluate_embeddings(embeddings)
    plot_embeddings(embeddings)
