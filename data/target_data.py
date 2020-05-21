#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/19 15:05
# Author  : zhoudong
# File    : target_data.py

import numpy as np
import matplotlib.pyplot as plt

class Target_Position:
    def __init__(self, N):
        self.s = np.zeros((4, N, 4), dtype=np.float32)       # 状态向量
        self.n = np.zeros((4, N))           # 4行N列         1 有真实目标

def target_form_nonoise(F, N, G, q_noise):
    """
    生成5个目标真实运动数据,
    :param F: 状态转移矩阵
    :param N: 时刻数目
    :param G: 过程噪声转移矩阵
    :param q_noise:  噪声标准差
    :return:
    """
    #target = Target_Position(N)
    target = []
    state = np.zeros((4, N, 4))                                  # 运动状态
    state_flag =  np.zeros((4, N))                      # 是否存在目标   1 存在    0 不存在

    state_flag[0, :8] = 1
    state_flag[1, 8:26] = 1
    state_flag[2, 12:38] = 1
    state_flag[3, 26:] = 1

    q_noise = np.reshape(q_noise, (-1, 1))             # 列向量

    # 目标1
    for i in range(N):
        if (i == 0):
            u = np.array([0, 2.6, 0, -1.2])
            state[0, i, :] = u
        elif i > 1 and i < 8:
            u = np.dot(F, u)
            state[0, i, :] = u


    # 目标2
    for i in range(N):
        if (i ==8):
            u = np.array([0, 0.6, 0, -2.1])
            state[1, i, :] = u
        elif i > 8 and i < 26:
            u = np.dot(F, u)
            state[1, i, :] = u

    # 目标3
    for i in range(N):
        if (i == 12):
            u = np.array([-10, 1.2, 0, 1.8])
            state[2, i, :] = u
        elif i > 12 and i < 38:
            u = np.dot(F, u)
            state[2, i, :] = u

    # target.s[2] = np.array(target.s[2])

    # 目标4
    for i in range(N):
        if (i == 26):
            u = np.array([0, 1.4, -5, -2.1])
            state[3, i, :] = u
        elif (i > 26):
            u = np.dot(F, u)
            state[3, i, :] = u


    plt.figure(1)
    plt.plot(state[0, :8, 0], state[0, :8, 2], color="blue", label="1")
    plt.plot(state[1, 8:26, 0], state[1, 8:26, 2], color="red", label="2")
    plt.plot(state[2, 12:38, 0], state[2, 12:38, 2], color="green", label="3")
    plt.plot(state[3, 26:, 0], state[3, 26:, 2], color="black", label="4")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend(loc="upper left")

    plt.figure(2)
    plt.plot(state[0, :8, 0], state[0, :8, 2], color="blue", label="1")
    plt.plot(state[1, 8:26, 0], state[1, 8:26, 2], color="red", label="2")
    plt.plot(state[2, 12:38, 0], state[2, 12:38, 2], color="green", label="3")
    plt.plot(state[3, 26:, 0], state[3, 26:, 2], color="black", label="4")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend(loc="upper left")

    target.append(state)
    target.append(state_flag)

    return target

def measures(H, r_noise, target, N, pd, r):
    """
    生成观测数据，在真实数据上加上噪声
    :param H:        观测矩阵
    :param r_noise:   量测噪声标准差
    :param target:    真实运动数据
    :param N:         目标数
    :param pd:        目标检测概率
    :param r:         杂波数目平均值
    :return:          返回量测数据 和 对于时刻的数目
    """
    # a = -100          # 范围
    # b = 100
    a = -100
    b = 100
    measuredatas = []
    nummeasures = []

    target_size = len(target[0])         # 目标数

    r_noise = np.reshape(r_noise, (-1, 1))        # 列向量

    for i in range(N):
        clutter = np.random.poisson(r)         # 泊松分布的随机数
        measuredata = []
        for j in range(clutter):
            y = a + (b-a)*np.random.rand(2,1)
            measuredata.append(y)
        # print("i: ", measuredata)
        measuredatas.append(measuredata)

    #numtruetargets = [len(measuredatas[i]) for i in range(N)]
    numtruetargets = []
    for i in range(N):
        count = 0
        for j in range(target_size):
            if target[1][j, i] == 1:
                count +=1
                s = np.random.rand(1)
                if s < pd:
                    y = np.dot(H, np.reshape(target[0][j, i, :], (np.shape(target[0][j, i, :])[0], 1))) + r_noise * np.random.randn(2,1)
                    measuredatas[i].append(y)
        nummeasures.append(len(measuredatas[i]))
        numtruetargets.append(count)

    plt.figure(1)
    for i in range(N):
        for j in range(nummeasures[i]):
            plt.scatter(measuredatas[i][j][0], measuredatas[i][j][1], alpha=0.4, linestyle="-")
    #
    #
    # plt.show()




    return measuredatas, nummeasures, numtruetargets