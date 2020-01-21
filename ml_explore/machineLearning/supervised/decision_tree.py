#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 Fisher. All rights reserved.
#   
#   文件名称：decision_tree.py
#   创 建 者：YuLianghua
#   创建日期：2020年01月20日
#   描    述：
#
#================================================================

import numpy as np

class Node(object):
    def __init__(self, left, right, rule):
        self.left = left
        self.right= right
        self.feature = rule[0]
        self.threshold = rule[1]

class Leaf(object):
    def __init__(self, value):
        self.value = value

class DT(object):
    def __init__(self, 
                 classifier=True,
                 max_depth=None,
                 n_feats=None,
                 criterion='entropy',
                 seed=None):
        '''
        A decision tree model for regression and classification problems.
        @param classifier: bool, Whether to treat target values as categorical (classifier =
                           True) or continuous (classifier = False). Default is True.
        @param max_depth: int or None, The depth at which to stop growing the tree.
                           If None, grow the tree until all leaves are pure. Default is None.
        @param n_feats: int, Specifies the number of features to sample on each split. If None,
                           use all features on each split. Default is None.
        @param criterion: string, {'mse', 'entropy', 'gini'}
        @param seed: int or None.
        '''
        if seed:
            np.random.seed(seed)

        self.depth = 0
        self.root = None

        self.n_feats = n_feats
        self.criterion = criterion
        self.classifier = classifier
        self.max_depth = max_depth

        if not classifier and criterion in ('gini', 'entropy'):
            raise ValueError(
                "{} is a valid criterion only when classifier = True.".format(criterion)
            )
        if classifier and criterion == 'mse':
            raise ValueError("`mse` is a valid criterion only when classifier = False.")

    def fit(self, X, Y):
        self.n_classes = max(Y) + 1 if self.classifier else None
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow(X, Y)

    def predict(self, X):
        '''
        use the trained decision tree to classify or predict the examples in `X`.
        '''
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_class_probs(self, X):
        '''
        use the trained decision tree to return the class probabilities for the 
        example in `X`.
        '''
        assert self.classifier, "`predict_class_probs` undefined for classifier = False"
        return np.array([self._traverse(x, self.root, prob=True) for x in X])

    def _grow(self, X, Y, cur_depth=0):
        '''
        build tree
        '''
        # if all labels are the same, return a leaf.
        if len(set(Y)) == 1:
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[Y[0]] = 1.0
            return Leaf(prob) if self.classifier else Leaf(Y[0])

        # if reached max_depth, return a leaf
        if cur_depth >= self.max_depth:
            v = np.mean(Y, axis=0)
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)
            return Leaf(v)

        cur_depth += 1
        self.depth = max(self.depth, cur_depth)

        N, M = X.shape
        feat_idxs = np.random.choice(M, self.n_feats, replace=False)

        # greedily select the best split according to `criterion`
        feat, thresh = self._segment(X, Y, feat_idxs)
        l = np.argwhere(X[:, feat] <= thresh).flatten()
        r = np.argwhere(X[:, feat] >  thresh).flatten()

        # grow the children that result from the split
        left = self._grow(X[l, :], Y[l], cur_depth)
        right= self._grow(X[r, :], Y[r], cur_depth)
        return Node(left, right, (feat, thresh))

    def _segment(self, X, Y, feat_idxs):
        '''
        find the optimal split rule (feature index and split threshold) for the 
        data according to 'self.criterion'.
        '''
        best_gain = -np.inf 
        split_idx, split_thresh = None, None

        for i in feat_idxs:
            vals = X[:, i]
            levels = np.unique(vals)
            thresholds = (levels[:-1] + levels[1:]) / 2 if len(levels)>1 else levels
            gains = np.array([self._impurity_gain(Y, t, vals) for t in thresholds])

            if gains.max() > best_gain:
                split_idx = i
                best_gain = gains.max()
                split_thresh = thresholds[gains.argmax()]

        return split_idx, split_thresh

    def _impurity_gain(self, Y, split_thresh, feat_values):
        '''
        compute the impurity gain associated with a given split.
        IG(split) == loss(parent) - weighted-avg[loss(left_child), loss(right_child)]
        '''
        if self.criterion == 'entropy':
            loss = entropy
        elif self.criterion == "gini":
            loss = gini
        elif self.criterion == "mse":
            loss = mse

        parent_loss = loss(Y)

        # generate split
        left = np.argwhere(feat_values <= split_thresh).flatten()
        right= np.argwhere(feat_values  > split_thresh).flatten()

        if len(left)==0 or len(right)==0:
            return 0

        # compute the weight avg. of the loss for the children
        n = len(Y)
        n_l, n_r = len(left), len(right)
        e_l, e_r = loss(Y[left]), loss(Y[right])
        child_loss = (n_l/n) * e_l + (n_r/n) * e_r

        # impurity gain is difference in loss before vs. after split
        ig = parent_loss - child_loss

        return ig

    def _traverse(self, X, node, prob=False):
        if isinstance(node, Leaf):
            if self.classifier:
                return node.value if prob else node.value.argmax()
            return node.value

        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        return self._traverse(X, node.right, prob)


def mse(y):
    '''
    mean squared error for decision tree(ie. mean) predictions.
    '''
    return np.mean( (y - np.mean(y))**2 )

def entropy(y):
    '''
    entropy of a label sequence.
    '''
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return -np.sum( [p * np.log2(p) for p in ps if p>0] )

def gini(y):
    '''
    gini impurity (local entropy) of a label sequence.
    '''
    hist = np.bincount(y)
    N = np.sum(hist)
    return 1 - sum( [(i/N)**2 for i in hist] )
