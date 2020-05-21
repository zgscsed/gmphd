#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/19 14:08
# Author  : zhoudong
# File    : gm_component.py

import numpy as np
import itertools

myfloat = np.float32

# 多元高斯组件，每个目标保存，方便计算
class GmphdComponent:
    def __init__(self, weight, loc, cov):
        """

        :param weight: 权值
        :param loc: 期望向量
        :param cov: 协方差矩阵
        """

        self.weight = np.float32(weight)
        # 使用 array
        self.loc = np.array(loc, dtype=np.float32, ndmin=2)
        self.cov = np.array(cov, dtype=np.float32, ndmin=2)
        # 调整行和列
        self.loc = np.reshape(self.loc, (np.size(self.loc), 1))         # 期望 列向量  n维
        self.n = np.size(loc)                                           # 状态期望维度
        self.cov = np.reshape(self.cov, (self.n, self.n))               # 协方差矩阵    维度n * n

        # 计算多元高斯分布的预先计算值
        self.part1 = (2*np.pi) ** (-self.n*0.5)
        self.part2 = np.linalg.det(self.cov) ** (-0.5)
        self.invcov = np.linalg.inv(self.cov)                           # 逆矩阵

    def dmvnorm(self, x):
        """
        计算在x处的多元高斯分布概率
        :param x: 状态向量
        :return:
        """
        x = np.array(x, dtype=myfloat)
        dev = x - self.loc

        part3 = np.exp(-0.5 * np.dot(np.dot(dev.T, self.invcov), dev))

        return self.part1 * self.part2 * part3

def dmvnorm(loc, cov, x):
    """
    不一定都是GmphdComponent ，因此单独写个函数计算概率
    :param lov: 期望
    :param cov: 协方差矩阵
    :param x: 状态向量
    :return:
    """
    loc = np.array(loc, dtype=myfloat)
    cov = np.array(cov, dtype=myfloat)
    x = np.reshape(x, (2,1))

    n = np.size(loc)

    part1 = (2 * np.pi) ** (-n * 0.5)
    #part1 = (2 * np.pi) ** (-0.5)
    part2 = np.linalg.det(cov) ** (-0.5)
    dev = x - loc
    part3 = np.exp(-0.5 * np.dot(np.dot(dev.T, np.linalg.inv(cov)), dev))

    return part1 * part2 * part3