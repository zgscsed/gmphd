#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/19 14:20
# Author  : zhoudong
# File    : demo.py

import matplotlib.pyplot as plt

from model.model import Model
from data import target_data

from util.gm_component import *
from util.ospa import opsa

# from phd import gmcphd
import gmcphd

# 仿真参数
N = 40           # 采样次数
M_number = 1      # 仿真次数
T_prun = 1e-5     # 合并阈值
U_merge = 5
J_max = 100       # 分量数
# ospa 参数
c_ospa = 100
p_ospa = 2

N_max = 20          # 存在目标最大数    cphd 的

model = Model()


# 生成真实数据
# target = target_form(F, N, G, q_noise)
target = target_data.target_form_nonoise(model.F, N, model.G, model.sigma_v)

# print(target[1][])

# 生成噪声数据
measuresdatas, nummeasures, numtruetargets= target_data.measures(model.H, model.sigma_r,
                                                                 target, N, model.P_D, model.lambda_c)


g = gmcphd.Gmcphd(model, N_max)
g.gmm.append(GmphdComponent(1, np.array([0,2.6,0,-1.2]), np.diag([1, 2, 1, 2])))

opsa_dist_cphd = []
number_cphd = []
for k in range(1, N):
    print("---%i-------------------------------------------------" % k)
    print("numtrue num is ", numtruetargets[k])

    g.update(measuresdatas[k])
    g.prune(T_prun, U_merge, J_max)
    # items = g.extractstatesusingintegral(1)
    # items = g.estimeate()
    items = g.extractstates()
    #items = g.extractstates(2.0)
    number_cphd.append(len(items))

    detecttarget = []              # 滤波器检测的目标
    plt.figure(2)
    for x in items:
        detecttarget.append(x.T)         # x 是列向量  之所以转置，是为了后面转为行向量保存
        plt.scatter(x[0], x[2], alpha=0.4, linestyle="-")

    detecttarget = np.array(detecttarget).reshape(-1, 4)

    truetarget = []
    for j in range(4):
        if target[1][j, k] == 1:
            truetarget.append(target[0][j, k, :])
    truetarget = np.array(truetarget)
    # 计算opsa距离
    temp = opsa(truetarget, detecttarget)
    opsa_dist_cphd.append(temp)

plt.figure(3)
plt.plot(range(1, N), numtruetargets[1:], marker='*',color="blue", label="true")
plt.plot(range(1, N), number_cphd, marker = '*', color="red", label="dete")
plt.legend(loc="upper left")

plt.figure(4)
plt.plot(range(1, N), opsa_dist_cphd, color="blue", label="opsa")


plt.show()


