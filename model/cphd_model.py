#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/14 13:24
# Author  : zhoudong
# File    : cphd_model.py

from g_tool import *
# spawn 参数
j_beta_k = 1           # 衍生目标数
w_beta_k = 0.05        # 权值
d_beta_k = np.array([0, 0, 0, 0]).reshape(4, 1)
F_beta_k = np.eye(4)
Q_beta_k = np.diag([0.01, 0.04, 0.01, 0.04])
class CphdModel:
    """
    cphd滤波的一些参数
    """
    def __init__(self):
        # 基本参数
        self.x_dim = 4                     # 状态向量的维度
        self.z_dim = 2                     # 观测向量的维度

        # 运动模型参数 cv
        self.T = 1                         # 采用时间
        self.F = np.array([[1, self.T, 0, 0],      # 状态转移矩阵
                           [0, 1, 0, 0],
                           [0, 0, 1, self.T],
                           [0, 0, 0, 1]])
        self.G = np.array([[self.T*self.T/2, 0],       #过程噪声转移矩阵
                           [self.T, 0],
                           [0, self.T*self.T/2],
                           [0, self.T]])
        self.sigma_v = np.array([[0.5, 0.1]], dtype=float)        # 过程噪声
        self.Q = np.dot(np.dot(self.G, np.dot(self.sigma_v.T, self.sigma_v) * np.eye(2)), self.G.T)  # 过程噪声协方差

        # 生存概率
        self.P_S = 0.99
        # 检测概率
        self.P_D = 0.9

        # 新生目标参数
        self.birthgmm = []
        self.birthgmm.append(GmphdComponent(0.01, [40, -1.5, 20, -1.5], np.diag([5, 0.01, 5, 0.01])))
        self.birthgmm.append(GmphdComponent(0.01, [50, -1.5, -50, 1], np.diag([5, 0.01, 5, 0.01])))
        self.birthgmm.append(GmphdComponent(0.01, [-60, 1.5, -40, 1.5], np.diag([5, 0.01, 5, 0.01])))

        # self.birthgmm.append(GmphdComponent(0.1, [0, 0, 0, 0], np.diag([5, 1, 5, 1])))
        # self.birthgmm.append(GmphdComponent(0.1, [-10, 0, 0, 0], np.diag([5, 1, 5, 1])))
        # self.birthgmm.append(GmphdComponent(0.1, [0, 0, -5, 0], np.diag([5, 1, 5, 1])))

        # 衍生目标spawn 参数
        self.j_beta_k = 1  # 衍生目标数
        self.w_beta_k = 0.05  # 权值
        self.d_beta_k = np.array([0, 0, 0, 0]).reshape(4, 1)
        self.F_beta_k = np.eye(4)
        self.Q_beta_k = np.diag([0.01, 0.04, 0.01, 0.04])

        # 观测模型参数
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])                       # 观测矩阵
        self.sigma_r =  np.array([[0.5,0.5]])                                 # 观测噪声
        self.R = np.dot(self.sigma_r.T, self.sigma_r)*np.eye(2)               # 观测噪声协方差

        # 杂波参数
        self.lambda_c = 5         # 杂波平均值  泊松分布
        self.range_c = np.array([[-70, 70], [-70, 70]])       # 均匀杂波区域
        self.pdf_c = 1 / np.power(self.range_c[0, 1] - self.range_c[0, 0], 2) # 均匀杂波密度


