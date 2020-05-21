#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/8 10:44
# Author  : zhoudong
# File    : example.py

from target_form import *
from mygmphd import *
from model.model_cv import *
import test
import time
#####################################
# 参数
T = 1      # 时间间隔
N = 100      # 目标数量
r = 3      # 杂波平均数
pd = 0.99   # 检测概率
ps = 0.99   # 存活概率
j_max = 100   # 高斯分量最大数
u_merg = 5    # 合并阈值
truncthresh = 1e-5     # 剪枝阈值

# 状态转移矩阵
F = np.array([[1, T, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, T],
              [0, 0, 0, 1]])
# 过程噪声转移矩阵
G = np.array([[T*T/2, 0],
              [T, 0],
              [0, T*T/2],
              [0, T]])
# 量测矩阵
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
q_noise = np.array([0.1, 0.1], dtype=float)        # 标准差
Q_noise = np.array([[0.01, 0],
                   [0, 0.01]], dtype=float)

Q = np.dot(np.dot(G, Q_noise), G.T)



r_noise = np.array([0.5,0.5], dtype=float)
R_noise = np.array([[0.25, 0],                     # 量测噪声方差
                   [0, 0.25]], dtype=float)

p = np.array([[10.0, 0, 0, 0],
              [0, 0.01, 0, 0],
              [0, 0, 10.0, 0],
              [0, 0, 0, 0.01]], dtype=np.float32)




#model = Model()

# 生成真实数据
# target = target_form(F, N, G, q_noise)
target = target_form_addnoise(F, N, G, q_noise)

# 生成噪声数据
measuresdatas, nummeasures, numtruetargets= measures(H, r_noise, target, N, pd, r)

birthgmm = []
#birthgmm.append(GmphdComponent(0.01, target.s[0, 0, :, :], p))
birthgmm.append(GmphdComponent(0.01, target.s[1, 20, :], np.diag([5, 0.01, 5, 0.01])))
birthgmm.append(GmphdComponent(0.01, target.s[2, 30, :], np.diag([5, 0.01, 5, 0.01])))
birthgmm.append(GmphdComponent(0.01, target.s[3, 40, :], np.diag([5, 0.01, 5, 0.01])))

g = Gmphd(birthgmm, ps, pd, F, Q, H, R_noise, r/(140*140))
#g = Gmphd(model)
g.gmm.append(GmphdComponent(1, target.s[0, 0, :], p))

#########################tests
#meas = [[-57.3359, -68.8453], [-17.8109, 40.3630], [-7.3800, -56.0526], [-38.7082, 49.1634]]
# meas = [[15.0283, 54.5661],[ -1.3040, -26.1089], [-39.3080, 49.0074]]
# g.update(meas)
# g.prune(truncthresh, u_merg, j_max)

filtertarget =[]           # 跟踪的目标
opsa_dist = []
for i in range(N):
    print("---%i-------------------------------------------------" % i)
    print("numtrue num is ", numtruetargets[i])
    start = time.time()


    g.update(measuresdatas[i])
    g.prune(truncthresh, u_merg, j_max)
    # items = g.extractstatesusingintegral(1)
    # items = g.estimeate()
    items = g.extractstates(1.0)
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
    opsa_dist.append(test.opsa(truetarget, detecttarget))
    end = time.time()

    print("time: " , end - start)





plt.figure(3)
plt.plot(range(N), numtruetargets, color="blue", label="true")
plt.plot(range(N), filtertarget, color="red", label="dete")
plt.legend(loc="upper left")

plt.figure(4)
plt.plot(range(N), opsa_dist, color="blue", label="opsa")


plt.show()










