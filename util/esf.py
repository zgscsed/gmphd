#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/15 16:21
# Author  : zhoudong
# File    : esf.py
import numpy as np
def esf(z):
    """
    函数的意思：
    输入一个向量
    z = [1,2,3]
    计算选择列表元素个数所有组合的和
    个数k 0,1,2,3
    example k = 2
    所有组合
    [1，2]  1*2 = 2
    [1，3]  1*3 = 3
    [2，3]  2*3 = 6
    和          = 11

    函数返回 [1, 6.0, 11.0, 6.0]
    :param z: 向量，观测值的似然，
    :return:
    """
    len_z = len(z)
    if len_z == 0:
        return 1

    F = np.zeros((2, len_z), dtype=np.float32)   #

    i_n = 0
    i_minus = 1

    for n in range(len_z):
        F[i_n, 0] = F[i_minus, 0] + z[n]
        for k in range(1, n+1):
            if k == n:
                F[i_n, k] = z[n]*F[i_minus, k-1]
            else:
                F[i_n, k] = F[i_minus, k] + z[n]*F[i_minus, k-1]
        temp =i_n
        i_n = i_minus
        i_minus = temp
    s = [1]
    s.extend(F[i_minus, :])
    return s


t = np.zeros((2,3))
t[0,0] = 1.0
t[0, 1] = 1
t[0,2] = 2
t[1,0] = 1.0
t[1, 1] = 1
t[1,2] = 2
# print(np.sum(t, axis=1))
# print(np.concatenate([t[:2], t[3:]]))
# print(sum(np.log(i) for i in range(1, 3)))