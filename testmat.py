#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/12 11:08
# Author  : zhoudong
# File    : testmat.py

from target_form import *
from mygmphd import *
import test

#####################################
# 参数
T = 1  # 时间间隔
N = 100  # 目标数量
r = 3  # 杂波平均数
pd = 0.9  # 检测概率
ps = 0.99  # 存活概率
j_max = 100  # 高斯分量最大数
u_merg = 5  # 合并阈值
truncthresh = 1e-5  # 剪枝阈值

# 状态转移矩阵
F = np.array([[1, T, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, T],
              [0, 0, 0, 1]])
# 过程噪声转移矩阵
G = np.array([[T * T / 2, 0],
              [T, 0],
              [0, T * T / 2],
              [0, T]])
# 量测矩阵
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
q_noise = np.array([0.1, 0.1], dtype=float)
Q_noise = np.array([[0.01, 0],
                    [0, 0.01]], dtype=float)

Q = np.dot(np.dot(G, Q_noise), G.T)

r_noise = np.array([0.5, 0.5], dtype=float)
R_noise = np.array([[0.25, 0],
                    [0, 0.25]], dtype=float)

p = np.array([[10.0, 0, 0, 0],
              [0, 1.0, 0, 0],
              [0, 0, 10.0, 0],
              [0, 0, 0, 1.0]], dtype=np.float32)

# 生成真实数据
# target = target_form(F, N, G, q_noise)
target = target_form_example1(F, N, G, q_noise)

# 生成噪声数据
# measuresdatas, nummeasures, numtruetargets = measures(H, r_noise, target, N, pd, r)

birthgmm = []
# birthgmm.append(GmphdComponent(0.01, target.s[0, 0, :, :], p))
birthgmm.append(GmphdComponent(0.01, target.s[1, 20, :], np.diag([5, 0.01, 5, 0.01])))
birthgmm.append(GmphdComponent(0.01, target.s[2, 30, :], np.diag([5, 0.01, 5, 0.01])))
birthgmm.append(GmphdComponent(0.01, target.s[3, 40, :], np.diag([5, 0.01, 5, 0.01])))

g = Gmphd(birthgmm, ps, pd, F, Q, H, R_noise, r / (140 * 140))
g.gmm.append(GmphdComponent(1, target.s[0, 0, :], p))

filtertarget = []  # 跟踪的目标
opsa_dist = []
measuresdatas = []
measuresdatas.append([[8.84818803895097,-39.7157744248680], [-29.7944497881248,49.8932365575465]])
measuresdatas.append([[8.84818803895097,-39.7157744248680], [-29.7944497881248,49.8932365575465]])

for i in range(N):
    print("---%i-------------------------------------------------" % i)
    #print("numtrue num is ", numtruetargets[i])

    g.update(measuresdatas[i])
    g.prune(truncthresh, u_merg, j_max)
    # items = g.extractstatesusingintegral(1)
    items = g.estimeate()
    # items = g.extractstates(2.0)
    filtertarget.append(len(items))

    detecttarget = []
    plt.figure(2)
    for x in items:
        detecttarget.append(x.T)
        # print("x", x)
        plt.scatter(x[0], x[2], alpha=0.4, linestyle="-")

    detecttarget = np.array(detecttarget).reshape(-1, 4)

    print("-------------------------")
    truetarget = []
    for j in range(5):
        if target.n[j, i] == 1:
            # truetarget[:, count] = target.s[j, i, :]
            truetarget.append(target.s[j, i, :])
            # print(target.s[j, i, :])
    truetarget = np.array(truetarget)

    opsa_dist.append(test.opsa(truetarget, detecttarget))

plt.figure(3)
#plt.plot(range(N), numtruetargets, color="blue", label="true")
plt.plot(range(N), filtertarget, color="red", label="dete")
plt.legend(loc="upper left")

plt.figure(4)
plt.plot(range(N), opsa_dist, color="blue", label="opsa")

plt.show()