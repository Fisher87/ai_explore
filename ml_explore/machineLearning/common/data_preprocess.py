#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 Fisher. All rights reserved.
#   
#   文件名称：data_preprocess.py
#   创 建 者：YuLianghua
#   创建日期：2019年12月23日
#   描    述：特征数据预处理:
#                       [1]. 数据标准化;
#                       [2]. 正则化;
#                       [3]. 特征离散化;
#                       [4]. 特征二值化;
#                       [5]. 类别特征编码;
#                       [6]. 缺失数据处理;
#                       [7]. 多项式特征创建;
#                       [8]. 自定义特征转换函数;
#
#             reference: https://blog.csdn.net/sinat_33761963/article/details/53433799
#================================================================

import numpy as np
from sklearn import preprocessing

x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])

# [1]. 数据标准化
#################################
def standardization(x):
    '''
    数据标准化:
            均值为0, 方差为1;
    '''
    x_scale = preprocessing.scale(x)
    # print(x_scale.mean(axis=0))   # 均值
    # print(x_scale.std(axis=0))    # 方差
    # out: array([[ 0.        , -1.22474487,  1.33630621],
    #             [ 1.22474487,  0.        , -0.26726124],
    #             [-1.22474487,  1.22474487, -1.06904497]])
    return x_scale

def standerhandle(x):
    """
    preprocessing这个模块还提供了一个实用类StandarScaler，
        它可以在训练数据集上做了标准转换操作之后，把相同的转换应用到测试训练集中。
        这是相当好的一个功能。可以对训练数据，测试数据应用相同的转换，以后有新的
        数据进来也可以直接调用，不用再重新把数据放在一起再计算一次了。
    """
    scaler = preprocessing.StandardScaler().fit(x)
    x_scale= scaler.transform(x)
    # x_new_scale = scaler.transform(x_new)
    return scaler, x_scale

def min_max_scale(x):
    '''
    规模化特征到一定的范围内, 一般情况下是在[0,1]之间,
    每个特征中的最小值变成了0，最大值变成了1;
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    x_minmax = min_max_scaler.fit_transform(x)
    # out:
    # array([[ 0.5       ,  0.        ,  1.        ],
    #    [ 1.        ,  0.5       ,  0.33333333],
    #    [ 0.        ,  1.        ,  0.        ]])
    return x_minmax

def max_abs_scale(x):
    '''
    原理和`min_max_scale`一样，只是数据会被规模化到[-1,1]之间.
    也就是特征中，所有数据都会除以最大值. 这个方法对那些已经中
    心化均值维0或者稀疏的数据有意义;
    '''
    max_abs_scaler = preprocessing.MaxAbsScaler()
    x_train_maxsbs = max_abs_scaler.fit_transform(x)
    # out:
    # array([[ 0.5, -1. ,  1. ],
    #    [ 1. ,  0. ,  0. ],
    #    [ 0. ,  1. , -0.5]])
    return x_train_maxsbs

'''
# a. 规模化稀疏数据:
#    如果对稀疏数据进行去均值的中心化就会破坏稀疏的数据结构。
#    虽然如此，我们也可以找到方法去对稀疏的输入数据进行转换，
#    特别是那些特征之间的数据规模不一样的数据。
# b. 规模化有异常值的数据:
#    如果你的数据有许多异常值，那么使用数据的均值与方差去做标准化就不行了。
#    可以使用robust_scale 和 RobustScaler这两个方法。
#    它会根据中位数或者四分位数去中心化数据。
'''

#################################

# [2].正则化
#################################

def normalize(x):
    x_norm = preprocessing.normalize(x, norm='l2')
    # out:
    # [[ 0.40824829 -0.40824829  0.81649658]
    #  [ 1.          0.          0.        ]
    #  [ 0.          0.70710678 -0.70710678]]
    return x_norm

def normalize_handle(x):
    normalizer = preprocessing.Normalizer().fit(x)
    x_norm = normalizer.transform(x)
    return normalizer, x_norm

#################################

# [3]. 特征离散化
#################################

#################################

# [4]. 特征的二值化 
#################################

def binarizer(x):
    '''
    默认是根据0来二值化，大于0的都标记为1，小于等于0的都标记为0。
    当然也可以自己设置这个阀值，只需传出参数threshold即可;
    '''
    binarizer = preprocessing.Binarizer().fix(x)
    # binarizer = preprocessing.Binarizer(threshold=1.5).fix(x)
    x_binary = binarizer.transform(x)
    # out:
    # array([[ 1.,  0.,  1.],
    #    [ 1.,  0.,  0.],
    #    [ 0.,  1.,  0.]])
    return binarizer, x_binary

#################################


# [5]. 类别特征编码  
#################################

x = np.array([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])

def category_enc(x):
    '''
    参数n_values，用来指明每个特征中的值的总个数,
    如n_values=[2,3,4], 表示特征1有2个值，特征2有3个值...;
    '''
    enc = preprocessing.OneHotEncoder().fit(x)
    # enc = preprocessing.OneHotEncoder(n_values=[2,3,4]).fit(x)
    x_onehot = enc.transform(x)
    return enc, x_onehot 

#################################


# [6]. 缺失数据处理  
#################################

x = [[1, 2], [np.nan, 3], [7, 6]]

def deficiency_pro(x):
    '''
    在scikit-learn的模型中都是假设输入的数据是数值型的，并且都是有意义的，
    如果有缺失数据是通过NAN，或者空值表示的话，就无法识别与计算了。
    '''
    imp = preprocessing.Imputer(missing_values="NaN", stategy="mean", axis=0)
    imp.fit(x)
    x_p = imp.transform(x)
    return imp, x_p

#################################


# [7]. 多项式特征创建 
#################################

def poly(x):
    '''
    有的时候线性的特征并不能做出美的模型，于是我们会去尝试非线性。
    非线性是建立在将特征进行多项式地展开上的。
    比如将两个特征 (X_1, X_2)，它的平方展开式便转换成5个特征:
         (1, X_1, X_2, X_1^2, X_1X_2, X_2^2), 第一列为bias.
    '''
    poly = preprocessing.PolynomialFeatures(2)
    # 可以自定义选择只要保留特征相乘的项
    # poly = PolynomialFeatures(degree=3, interaction_only=True)
    x_poly = poly.fit_transform(x)
    return x_poly

#################################


# [8]. 自定义特征转换函数 
#################################

def define_fuc(x):
    '''
    通俗的讲，就是把原始的特征放进一个函数中做转换，这个函数出来的值作为新的特征;
    '''
    fuc = np.log1p
    transformer = preprocessing.FunctionTransformer(fuc)
    x_t = transformer.transform(x)
    return x_t
