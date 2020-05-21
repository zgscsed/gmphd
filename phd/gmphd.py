#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/19 14:13
# Author  : zhoudong
# File    : gmphd.py

import numpy as np
from copy import deepcopy

from util.gm_component import *

myfloat = np.float32

class Gmphd:
    def __init__(self, model):
        """

        :param birthgmm:   GMM 的新生目标
        :param survival:   存活概率
        :param detection:  检测概率
        :param f:          状态转移矩阵
        :param q:          过程噪声协方差
        :param h:          观测矩阵
        :param r:          观察噪声协方差
        :param clutter:    杂波强度            杂波平均数目/区域面积
        """
        # self.gmm = []
        # self.birthgmm = birthgmm
        # self.survival = myfloat(survival)  # p_{s,k}(x) in paper
        # self.detection = myfloat(detection)  # p_{d,k}(x) in paper
        # self.f = np.array(f, dtype=myfloat)  # F_k-1 in paper
        # self.q = np.array(q, dtype=myfloat)  # Q_k-1 in paper
        # self.h = np.array(h, dtype=myfloat)  # H_k in paper
        # self.r = np.array(r, dtype=myfloat)  # R_k in paper
        # self.clutter = myfloat(clutter)      # KAU in paper

        self.gmm = []
        self.birthgmm = model.birthgmm  # gmm 新生目标
        self.survival = model.P_S  # p_{s,k}(x) in paper
        self.detection = model.P_D  # p_{d,k}(x) in paper
        self.f = model.F  # F_k-1 in paper
        self.q = model.Q  # Q_k-1 in paper
        self.h = model.H  # H_k in paper
        self.r = model.R  # R_k in paper
        self.clutter = model.lambda_c * model.pdf_c  # KAU in paper
        self.model = model

    def update(self, obs):
        """
        预测 更新，
        :param obs: 当前时刻的观察值
        :return:
        """
        #############################################
        # step1 - prediction for birth targets
        born = [deepcopy(comp) for comp in self.birthgmm]

        # # 衍生目标计算
        # spawned = [GmphdComponent(
        #     comp.weight * self.model.w_beta_k,
        #     self.model.d_beta_k + np.dot(self.model.F_beta_k, comp.loc),
        #     self.model.Q_beta_k + np.dot(np.dot(self.model.F_beta_k, comp.cov), self.model.F_beta_k.T)
        # ) for comp in self.gmm]

        # spawned = GmphdComponent(
        #     0.01,
        #     np.array([-3.5, 0, -23.5, -1.5]),
        #     np.diag([5, 0.01, 5, 0.01])
        # )
        #
        # spawned = [GmphdComponent(
        #     spawned.weight * self.survival,
        #     np.dot(self.f, spawned.loc),
        #     self.q + np.dot(np.dot(self.f, spawned.cov), self.f.T)
        # )]



        ##############################################
        # step2 - prediction for existing targets
        update = [GmphdComponent(
            self.survival * comp.weight,
            np.dot(self.f, comp.loc),
            self.q + np.dot(np.dot(self.f, comp.cov), self.f.T)
        )for comp in self.gmm]

        # print("updata len is", len(update))
        # print("spawned len is", len(spawned))
        # print("updata len is", len(born))

        predicted = born + spawned + update           # 存活目标 衍生目标 新生目标

        #######################################
        # Step 3 - construction of PHD update components
        # 利用卡尔曼滤波计算均值、权值和协方差矩阵
        # nu 从状态空间转为观测空间
        # s 协方差
        nu = [np.dot(self.h, comp.loc) for comp in predicted]
        s = [self.r + np.dot(np.dot(self.h, comp.cov), self.h.T) for comp in predicted]
        # k 增益   pkk 更新协方差
        k = [np.dot(np.dot(comp.cov, self.h.T), np.linalg.inv(s[index]))
             for index, comp in enumerate(predicted)]
        pkk = [np.dot(np.eye(len(k[index])) - np.dot(k[index], self.h), comp.cov)
               for index, comp in enumerate(predicted)]

        #######################################
        # Step 4 - update using observations
        # The 'predicted' components are kept, with a decay
        newgmm = [GmphdComponent( comp.weight * (1.0 - self.detection),comp.loc, comp.cov)
                  for comp in predicted]                                    # 漏检目标

        # predicted 根据 obs 更新
        for anobs in obs:
            anobs = np.array(anobs).reshape((2, 1))
            newgmmpartial = []
            for j, comp in enumerate(predicted):
                newgmmpartial.append(GmphdComponent(
                    self.detection * comp.weight * dmvnorm(nu[j], s[j], anobs),
                    comp.loc + np.dot(k[j], anobs - nu[j]),
                    pkk[j]
                ))

            # The Kappa thing (clutter and reweight)
            weightsum = sum(newcomp.weight for newcomp in newgmmpartial)
            reweighter = 1.0 / (self.clutter + weightsum)

            for newcomp in newgmmpartial:
                newcomp.weight *= reweighter

            newgmm.extend(newgmmpartial)

        self.gmm = newgmm

    def prune(self, truncthresh=1e-6, mergethresh=0.01, maxcomponents=100):
        """

        :param truncthresh:      权值阈值
        :param mergethresh:      合并阈值
        :param maxcomponents:    高斯分量最大数
        :return:
        """
        # Truncation is easy
        weightsums = [sum(comp.weight for comp in self.gmm)]
        sourcegmm = list(filter(lambda comp : comp.weight > truncthresh, self.gmm))
        weightsums.append(sum(comp.weight for comp in sourcegmm))
        origlen = len(self.gmm)
        trunclen = len(sourcegmm)

        # 计算新的高斯分量
        newgmm = []
        while len(sourcegmm) > 0:
            # 找出最大权值的分量
            windex = np.argmax([comp.weight for comp in sourcegmm])
            weightiest = sourcegmm[windex]                            # 最大权重
            sourcegmm = sourcegmm[:windex] + sourcegmm[windex+1:]

            # 合并高斯分量，  分布接近的合并
            distance = [float(np.dot(np.dot((comp.loc-weightiest.loc).T, np.linalg.inv(comp.cov)),
                                     comp.loc-weightiest.loc)) for comp in sourcegmm]
            dosubsume = np.array([dist <= mergethresh for dist in distance])            # 小于预设值的合并
            subsumed = [weightiest]

            # 将要合并的高斯分量从总的gmm中分离
            if np.any(dosubsume):
                subsumed.extend(list(np.array(sourcegmm)[dosubsume]))
                sourcegmm = list(np.array(sourcegmm)[~dosubsume])

            # 计算合并的高斯分量的值
            # w = w1+ w2 + ....        新权值为权值和
            # m = (w1*x1 + w2*x2 +...) / w      新期望
            # p = （(w1*(p1 + (max - x1)(max-x1).T))+.. ）/ w  新协方差
            aggweight = sum(comp.weight for comp in subsumed)      #权值和
            newloc = sum(comp.weight * comp.loc for comp in subsumed) / aggweight
            newcomp = GmphdComponent(aggweight,
                                     newloc,
                                     sum(comp.weight * (comp.cov + (newloc - comp.loc) *
                                         (newloc - comp.loc).T) for comp in subsumed ) / aggweight)
            newgmm.append(newcomp)

        # 按权重降序排序，选择前maxcomponents 个分量
        newgmm = sorted(newgmm, key=lambda comp:comp.weight)
        newgmm.reverse()
        self.gmm = newgmm[:maxcomponents]
        weightsums.append(sum(comp.weight for comp in newgmm))
        weightsums.append(sum(comp.weight for comp in self.gmm))

        print("prune(): %d -> %d -> %d -> %d" % (origlen, trunclen, len(newgmm), len(self.gmm)))
        print("prune(): weightsums %g -> %g -> %g -> %g" % (weightsums[0], weightsums[1], weightsums[2], weightsums[3]))
        # 剪枝不能减少权值和，需要重新规范化
        weightnorm = weightsums[0] / weightsums[3]

        for comp in self.gmm:
            comp.weight *= weightnorm


    def extractstatesusingintegral(self, bias = 1.0):
        """
        提取多个目标状态，返回
        先计算一共有多少目标   numtoadd
        每次找到权值最大的分量，将loc 保存在状态列表里，
        将该分量的权值 - 1
        将总个数 - 1
        :param bias: 偏差   不太理解
        :return:
        """
        numtoadd = int(round(float(bias) * sum(comp.weight for comp in self.gmm)))          # 偏差 * 权值和
        print("bias is %g, numtoadd is %i" % (bias, numtoadd))

        items = []
        peaks = [{'loc':comp.loc, 'weight' : comp.weight} for comp in self.gmm]
        while numtoadd > 0:
            windex = 0
            wsize = 0
            for which, peak in enumerate(peaks):
                if peak['weight'] > wsize:
                    windex = which
                    wsize = peak['weight']
            # add the winner
            items.append(deepcopy(peaks[windex]['loc']))
            peaks[windex]['weight'] -= 1.0
            numtoadd -= 1
        return items

    def extractstates(self, bias=1.0):
        """
        选择权重大于0.5 的分量的状态
        :param bias:
        :return:
        """
        items = []
        print("weights:")
        print([np.round(comp.weight, 7)
        for comp in self.gmm])
        for comp in self.gmm:
            val = comp.weight * float(bias)
            if val > 0.5:
                for _ in range(int(np.round(val))):
                    items.append(deepcopy(comp.loc))
        return items