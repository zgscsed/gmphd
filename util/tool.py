#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/19 16:08
# Author  : zhoudong
# File    : tool.py

import numpy as np
import itertools
# 阶乘
def factorial(n):
    f = 1
    for i in range(1, n + 1):
        f *= i
    return f

# 计算k 次发生的概率， 泊松分布
def poisspdf(k, lambda_c):
    return np.power(np.e, -lambda_c)*np.power(lambda_c, k)/factorial(k)

# 初等对称函数

def delta(L, j):
    if j==0:
        y = 1
    else:
        tmp = list(itertools.combinations(L, j))        # 输出组合
        tem_shape = np.shape(tmp)
        temp = np.ones(tem_shape[0])
        for i in range(tem_shape[0]):
            for j in range(tem_shape[1]):
                temp[i] = temp[i] * tmp[i][j]
        y = sum(temp)
    return y