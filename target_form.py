#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/7 16:28
# Author  : zhoudong
# File    : target_form.py

import numpy as np
import matplotlib.pyplot as plt

class Target_Position:
    def __init__(self, N):
        self.s = np.zeros((5, N, 4), dtype=np.float32)       # 状态向量
        self.n = np.zeros((5, N))           # 4行N列         1 有真实目标

def target_form_nonoise(F, N, G, q_noise):
    """
    生成5个目标真实运动数据,
    :param F: 状态转移矩阵
    :param N: 时刻数目
    :param G: 过程噪声转移矩阵
    :param q_noise:  噪声标准差
    :return:
    """
    target = Target_Position(N)

    target.n[0, :60] = 1
    target.n[1, 20:80] = 1
    target.n[2, 30:] = 1
    target.n[3, 40:] = 1
    target.n[4, 50:80] = 1

    q_noise = np.reshape(q_noise, (-1, 1))             # 列向量

    # 目标1 1-60秒存在
    for i in range(N):
        if (i == 0):
            u = np.array([-40, 1.5, 50, -1])
            target.s[0, i, :] = u
        elif i > 0 and i < 60:
            u = np.dot(F, u)
            #print(u.shape)
            target.s[0, i, :] = u


    # 目标2和5
    for i in range(N):
        if (i == 20):
            u = np.array([40, -1.5, 20, -1.5])
            target.s[1, i, :] = u
        elif i > 20 and i < 80:
            u = np.dot(F, u)
            target.s[1, i, :] = u

        if i == 50:
            v = np.array([u[0], 0, u[2], -1.5])
            target.s[4, i, :] = v
        elif i > 50 and i < 80:
            v = np.dot(F, v)
            target.s[4, i, :] = v

    # 目标3
    for i in range(N):
        if (i == 30):
            u = np.array([50, -1.5, -50, 1])
            target.s[2, i, :] = u
        elif i > 30:
            u = np.dot(F, u)
            target.s[2, i, :] = u

    # target.s[2] = np.array(target.s[2])

    # 目标4
    for i in range(N):
        if (i == 40):
            u = np.array([-60, 1.5, -40, 1.5])
            target.s[3, i, :] = u
        elif (i > 40):
            u = np.dot(F, u)
            target.s[3, i, :] = u

    plt.figure(1)
    plt.plot(target.s[0, :60, 0], target.s[0, :60, 2], color="blue", label="1")
    plt.plot(target.s[1, 20:80, 0], target.s[1, 20:80, 2], color="red", label="2")
    plt.plot(target.s[2, 30:, 0], target.s[2, 30:, 2], color="green", label="3")
    plt.plot(target.s[3, 40:, 0], target.s[3, 40:, 2], color="black", label="4")
    plt.plot(target.s[4, 50:80, 0], target.s[4, 50:80, 2], color="yellow", label="5")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend(loc="upper left")

    plt.figure(2)
    plt.plot(target.s[0, :60, 0], target.s[0, :60, 2], color="blue", label="1")
    plt.plot(target.s[1, 20:80, 0], target.s[1, 20:80, 2], color="red", label="2")
    plt.plot(target.s[2, 30:, 0], target.s[2, 30:, 2], color="green", label="3")
    plt.plot(target.s[3, 40:, 0], target.s[3, 40:, 2], color="black", label="4")
    plt.plot(target.s[4, 50:80, 0], target.s[4, 50:80, 2], color="yellow", label="5")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend(loc="upper left")

    return target

def target_form_addnoise(F, N, G, q_noise):
    """
    生成5个目标真实运动数据,     加入了过程噪声
    :param F:      状态转移矩阵
    :param N:         数量
    :return:
    """
    target = Target_Position(N)

    target.n[0, :60] = 1
    target.n[1, 20:80] = 1
    target.n[2, 30:] = 1
    target.n[3, 40:] = 1
    target.n[4, 50:80] = 1

    q_noise = np.reshape(q_noise, (-1, 1))             # 列向量

    # 目标1 1-60秒存在
    for i in range(N):
        if (i == 0):
            u = np.array([-40, 1.5, 50, -1])
            target.s[0, i, :] = u
        elif i > 0 and i < 60:
            u = np.dot(F, u).reshape((4,1))  + np.dot(G, q_noise * np.random.randn(2, 1))
            #print("u", u.shape)
            target.s[0, i, :] = u.T
    # target.s[0] = np.array(target.s[0])
    # print(target.s[0])

    # 目标2和5
    for i in range(N):
        if (i == 20):
            u = np.array([40, -1.5, 20, -1.5])
            target.s[1, i, :] = u
        elif i > 20 and i < 80:
            u = np.dot(F, u).reshape((4, 1)) + np.dot(G, q_noise * np.random.randn(2, 1))
            target.s[1, i, :] = u.T

        if i == 50:
            v = np.array([u[0], 0, u[2], -1.5])
            target.s[4, i, :] = v
        elif i > 50 and i < 80:
            v = np.dot(F, v)
            target.s[4, i, :] = v.T

    # 目标3
    for i in range(N):
        if (i == 30):
            u = np.array([50, -1.5, -50, 1])
            target.s[2, i, :] = u
        elif i > 30:
            u = np.dot(F, u).reshape((4, 1)) + np.dot(G, q_noise * np.random.randn(2, 1))
            target.s[2, i, :] = u.T

    # target.s[2] = np.array(target.s[2])

    # 目标4
    for i in range(N):
        if (i == 40):
            u = np.array([-60, 1.5, -40, 1.5])
            target.s[3, i, :] = u
        elif (i > 40):
            u = np.dot(F, u).reshape((4, 1)) + np.dot(G, q_noise * np.random.randn(2, 1))
            target.s[3, i, :] = u.T

    plt.figure(1)
    plt.plot(target.s[0, :60, 0], target.s[0, :60, 2], color="blue", label="1")
    plt.plot(target.s[1, 20:80, 0], target.s[1, 20:80, 2], color="red", label="2")
    plt.plot(target.s[2, 30:, 0], target.s[2, 30:, 2], color="green", label="3")
    plt.plot(target.s[3, 40:, 0], target.s[3, 40:, 2], color="black", label="4")
    plt.plot(target.s[4, 50:80, 0], target.s[4, 50:80, 2], color="yellow", label="5")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend(loc="upper left")

    plt.figure(2)
    plt.plot(target.s[0, :60, 0], target.s[0, :60, 2], color="blue", label="1")
    plt.plot(target.s[1, 20:80, 0], target.s[1, 20:80, 2], color="red", label="2")
    plt.plot(target.s[2, 30:, 0], target.s[2, 30:, 2], color="green", label="3")
    plt.plot(target.s[3, 40:, 0], target.s[3, 40:, 2], color="black", label="4")
    plt.plot(target.s[4, 50:80, 0], target.s[4, 50:80, 2], color="yellow", label="5")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend(loc="upper left")

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
    a = -70
    b = 70
    measuredatas = []
    nummeasures = []

    target_size = len(target.s)         # 目标数

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
            if target.n[j, i] == 1:
                count +=1
                s = np.random.rand(1)
                if s < pd:
                    y = np.dot(H, np.reshape(target.s[j, i, :], (np.shape(target.s[j, i, :])[0], 1))) + r_noise * np.random.randn(2,1)
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





















