#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/11 11:15
# Author  : zhoudong
# File    : test.py

import numpy as np

class matchEm:
    def __init__(self, weight):
        """
        最小权匹配算法
        :param weight: 权矩阵  n * m
        """
        self.weight = -weight
        self.x_len = weight.shape[0]
        self.y_len = weight.shape[1]

        self.visit_x = np.zeros(self.x_len, dtype=np.int)
        self.visit_y = np.zeros(self.y_len, dtype=np.int)
        self.weight_x = np.max(self.weight, axis=1)
        self.weight_y = np.zeros(self.y_len)
        self.match_x = np.zeros(self.x_len)-1
        self.match_x = self.match_x.astype(np.int32)
        self.match_y = (np.zeros(self.y_len)-1).astype(np.int32)

        self.min_x = np.min(self.weight, axis=1)

        self.mind_ = np.inf

    def match(self):
        count = 0
        for i in range(self.x_len):
            while True:
                # 每次初始化
                self.visit_x = np.zeros(self.x_len, dtype=np.int)
                self.visit_y = np.zeros(self.y_len, dtype=np.int)
                self.mind_ = np.inf
                if self.dfs(i):
                    break       # 匹配了， 就下个继续
                # 更新顶标
                if self.mind_ == np.inf:
                    break
                for j in range(self.x_len):
                    if self.visit_x[j] == 1:
                        self.weight_x[j] -= self.mind_
                if self.weight_x[i] < self.min_x[i]:
                    break
                for j in range(self.y_len):
                    if self.visit_y[j] == 1:
                        self.weight_y[j] += self.mind_
         # 输出匹配
        matching = np.zeros(self.weight.shape)  # 匹配完成， 匹配对于的值为1， 否则为0
        cost = 0
        for j in range(self.x_len):
            matching[j][self.match_x[j]] = 1
            cost -= self.weight[j][self.match_x[j]]

        return matching, cost



    def dfs(self, index):
        self.visit_x[index] = 1

        # 遍历y
        for i in range(self.y_len):
            # 如果没有访问
            if self.visit_y[i] == 0:
                mind = self.weight_x[index] + self.weight_y[i] - self.weight[index, i]
                if np.abs(mind) < 0.0001:
                    # mind 为0， 加入增广路
                    self.visit_y[i] = 1
                    # 没有匹配或已经匹配了， 重新匹配
                    if self.match_y[i] == -1 or self.dfs(self.match_y[i]):
                        self.match_x[index] = i
                        self.match_y[i] = index
                        return True
                else:
                    self.mind_ = min(mind, self.mind_)
        return False

"""
x, y 都是列向量

"""
def opsa(x, y):
    """
    计算opsa 距离
    :param x:   是 n 行
    :param y:    m 行
    :return:
    """
    n = x.shape[0]
    m = y.shape[0]
    print("test ---------------------")
    # print(x, y)

    if n==0 and m==0 :
        return 0
    if n==0 or m==0:
        return 100

    xx = np.vstack([x for i in range(m)])  # 行拼接
    # print(xx)
    yy = np.vstack([y[i, :] for i in range(m) for j in range(n)]).reshape(n*m, 4)  # 行重复 1 2 3 ->  1 1 2 2 3 3
    # print
    # print(yy-xx)
    d = np.sqrt(np.square(np.sum(yy - xx, axis=1))).reshape((n, m))
    d = np.power(np.minimum(d, 100), 2)
    # print(d)

    if n > m:
        d = d.T
    matchem = matchEm(d)
    print("d", d)
    match, cost = matchem.match()
    print("match", match)

    dist = (1 / max(n, m)) * (np.power(100, 2) * abs(m - n) + cost)
    dist = np.power(dist, 1 / 2)
    # print(dist)

    return dist

# x = np.array([[-9.9945, 1.5049, 29.9482, -1.0092]]).reshape((4, 1))
# x1 = np.array([[40.5826, -1.5000, 20.0616, -1.5000]]).reshape((4, 1))
# x = np.concatenate([x, x1], axis=1)
#
# n = x.shape[1]
#
#
# y = np.array([[-10, 1.500, 30, -1], [40, -1.50, 20, -1.50]]).T
# m = y.shape[1]
#
# # 注意： python 行优先
# # 方法一
# # 将x, y扩展， 因为n*m
# xx = np.concatenate([x for i in range(m)], axis=1)       # 行拼接
# yy = np.vstack([y[:, i] for i in range(m) for j in range(n)]).reshape(4, n*m).T       # 每列重复 1 2 3 ->  1 1 2 2 3 3
# d = np.sqrt(np.square(np.sum(yy - xx, axis=0))).reshape((n, m))
# d = np.power(np.minimum(d, 100), 2)
#
# print(d)
# matchem = matchEm(d)
# match, cost = matchem.match()
# print(match, cost)
# cost = (1/max(n,m))*(np.power(100, 2) * abs(m-n) + cost)
# cost = np.power(cost, 1/2)
# print(cost)
#dist = (q/max(n,m))*( (np.power(100, 2))*np.abs(m-n) + cost))
# 方法二
# d = np.zeros((2,2), dtype=np.float32)
# for i in range(2):
#     # 这里求的距离 loc 每个元素之差的和
#     tem = y[:, i]
#     for j in range(1):
#         tem = np.vstack((tem, y[:, i]))
#     tem = tem.T
#     d[:, i] = np.sqrt(np.square( np.sum( tem - x, axis=0) ))   # d[:, i] 是一维向量   sum axis=0 列求和
#
#
d = np.array([[6.3541263e-01,2.2755642e+03,2.4066191e+03],
 [3.1058872e-01,9.6220039e+03,1.0000000e+04]])
matchem = matchEm(d)
print(matchem.match())


