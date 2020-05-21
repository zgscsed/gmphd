#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/14 15:39
# Author  : zhoudong
# File    : demo_gmcphd.py

import gmcphd
from g_tool import *
import target_form
from model import cphd_model
import time
import matplotlib.pyplot as plt
import test

# 仿真参数
N = 100           # 采样次数
M_number = 1      # 仿真次数
T_prun = 1e-5     # 合并阈值
U_merge = 5
J_max = 100       # 分量数
N_max = 20        # 最大基数
c_ospa = 100
p_ospa = 2
N_max = 20


p = np.array([[10, 0, 0, 0],
              [0, 0.1, 0, 0],
              [0, 0, 10, 0],
              [0, 0, 0, 0.1]], dtype=np.float32)

model = cphd_model.CphdModel()


# 生成真实数据
# target = target_form(F, N, G, q_noise)
target = target_form.target_form_nonoise(model.F, N, model.G, model.sigma_v)

# 生成噪声数据
measuresdatas, nummeasures, numtruetargets= target_form.measures(model.H, model.sigma_r,
                                                                 target, N, model.P_D, model.lambda_c)

birthgmm = []
#birthgmm.append(GmphdComponent(0.01, target.s[0, 0, :, :], p))
birthgmm.append(GmphdComponent(0.01, target.s[1, 20, :], np.diag([5, 0.01, 5, 0.01])))
birthgmm.append(GmphdComponent(0.01, target.s[2, 30, :], np.diag([5, 0.01, 5, 0.01])))
birthgmm.append(GmphdComponent(0.01, target.s[3, 40, :], np.diag([5, 0.01, 5, 0.01])))

g = gmcphd.Gmcphd(model, N_max)
#g.gmm.append(GmphdComponent(1, np.array([0,2.6,0,-1.2]), p))
g.gmm.append(GmphdComponent(1, target.s[0, 0, :], p))
#########################tests
#meas = [[-57.3359, -68.8453], [-17.8109, 40.3630], [-7.3800, -56.0526], [-38.7082, 49.1634]]
# meas = [[15.0283, 54.5661],[ -1.3040, -26.1089], [-39.3080, 49.0074]]
# g.update(meas)
# g.prune(truncthresh, u_merg, j_max)
# meas = [[-53.1187, -43.0528], [95.7827, 91.8060], [2.9445, -1.7873]]
# g.update(meas)
# g.prune(T_prun, U_merge, J_max)

filtertarget =[]           # 跟踪的目标
opsa_dist = []
for i in range(N):
    print("---%i-------------------------------------------------" % i)
    print("numtrue num is ", numtruetargets[i])
    start = time.time()


    g.update(measuresdatas[i])
    g.prune(T_prun, U_merge, J_max)
    # items = g.extractstatesusingintegral(1)
    # items = g.estimeate()
    items = g.extractstates()
    #items = g.extractstates(2.0)
    filtertarget.append(len(items))

    detecttarget = []              # 滤波器检测的目标
    plt.figure(2)
    for x in items:
        detecttarget.append(x.T)         # x 是列向量  之所以转置，是为了后面转为行向量保存
        plt.scatter(x[0], x[2], alpha=0.4, linestyle="-")

    detecttarget = np.array(detecttarget).reshape(-1, 4)

    truetarget = []
    for j in range(5):
        if target.n[j, i] == 1:
            truetarget.append(target.s[j, i, :])
    truetarget = np.array(truetarget)
    # 计算opsa距离
    temp = test.opsa(truetarget, detecttarget)
    opsa_dist.append(temp)
    end = time.time()

    print("time: " , end - start)





plt.figure(3)
plt.plot(range(N), numtruetargets, color="blue", label="true")
plt.plot(range(N), filtertarget, color="red", label="dete")
plt.legend(loc="upper left")

plt.figure(4)
plt.plot(range(N), opsa_dist, color="blue", label="opsa")


plt.show()

