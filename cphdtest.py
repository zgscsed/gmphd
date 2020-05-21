#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/18 14:45
# Author  : zhoudong
# File    : cphdtest.py

from copy import deepcopy
from model.cphd_model import *


class Gmcphd:
    def __init__(self, model, N_max):
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
        self.gmm = []
        self.birthgmm = model.birthgmm             # gmm 新生目标
        self.survival = model.P_S                  # p_{s,k}(x) in paper
        self.detection = model.P_D                 # p_{d,k}(x) in paper
        self.f = model.F                           # F_k-1 in paper
        self.q = model.Q                           # Q_k-1 in paper
        self.h = model.H                           # H_k in paper
        self.r = model.R                           # R_k in paper
        self.clutter = model.lambda_c * model.pdf_c     # KAU in paper
        self.model = model

        # 势分布相关参数
        self.N_max = N_max                          # 最大势
        self.filter_pn = np.zeros(self.N_max+1)      # 势分布
        self.filter_pn[1] = 1

        self.M = np.zeros((self.N_max+1, self.N_max+1))
        for i in range(self.N_max+1):
            for j in range(self.N_max+1):
                tmp = 0
                for m in range(min(i,j)+1):
                    tmp = tmp + poisspdf(i-m, 0.1)*factorial(j)/factorial(j-m)/factorial(m)*\
                        np.power(1-self.survival, j-m)*np.power(self.survival, m)
                self.M[i, j] = tmp


    def update(self, obs):
        """
        预测 更新，
        :param obs: 当前时刻的观察值
        :return:
        """

        #############################################
        # step1 - prediction for birth targets
        birth = [deepcopy(comp) for comp in self.birthgmm]

        # 衍生目标计算
        spawned = [GmphdComponent(
            comp.weight * w_beta_k,
            d_beta_k + np.dot(F_beta_k, comp.loc),
            Q_beta_k + np.dot(np.dot(F_beta_k, comp.cov), F_beta_k.T)
        ) for comp in self.gmm]

        ##############################################
        # step2 - prediction for existing targets
        updated = [GmphdComponent(
            self.survival * comp.weight,
            np.dot(self.f, comp.loc),
            self.q + np.dot(np.dot(self.f, comp.cov), self.f.T)
        ) for comp in self.gmm]

        predicted = birth + updated

        ########################################################
        # 预测势分布
        pred_pn = np.zeros(self.N_max+1)
        for n in range(self.N_max+1):
            tmp = 0
            for n1 in range(self.N_max+1):
                tmp = tmp + self.filter_pn[n1]*self.M[n,n1]
            pred_pn[n] = tmp
        pred_n = sum(pred_pn[i]*i for i in range(self.N_max+1))

        # 更新势分布
        ############################################################################################
        # cardinalitu prediction
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

        # predicted 根据 obs 更新
        qz_temp = []  # 保存
        newobsgmm = []
        for anobs in obs:
            anobs = np.array(anobs).reshape((2, 1))
            newgmmpartial = []
            qz_j = 0
            for j, comp in enumerate(predicted):
                weight_temp = self.detection * comp.weight * dmvnorm(nu[j], s[j], anobs)
                qz_j += weight_temp
                newgmmpartial.append(GmphdComponent(
                    weight_temp,
                    comp.loc + np.dot(k[j], anobs - nu[j]),
                    pkk[j]
                ))
            qz_j = qz_j/pred_n
            qz_temp.append(qz_j)
            newobsgmm.append(newgmmpartial)

        qz_temp = np.array(qz_temp)
        qz_temp = qz_temp.flatten()

        # 根据公式更新
        alpha = np.zeros(self.N_max+1)
        for j in range(self.N_max+1):
            for n in range(j, self.N_max+1):
                alpha[j] = alpha[j] + factorial(n)/factorial(n-j)*pred_pn[n]*np.power(1-self.detection, n-j)

        obs_len = len(obs)
        beta = np.zeros(obs_len+1)
        for j in range(obs_len+1):
            beta[j] = poisspdf(obs_len-j, self.model.lambda_c)*factorial(obs_len-j)/factorial(obs_len)/np.power(self.clutter,j)

        L_z =0
        for j in range(obs_len+1):
            L_z = L_z + alpha[j]*beta[j]*delta(qz_temp, j)

        for j in range(obs_len+1):
            temp = delta(qz_temp, j)

        L_nd = 0
        for j in range(obs_len+1):
            L_nd = L_nd + alpha[j+1]*beta[j]*delta(qz_temp, j)

        L_nd = L_nd/pred_n

        # 更新目标
        newgmm = [GmphdComponent( comp.weight * (1.0 - self.detection) * L_nd/L_z, comp.loc, comp.cov)
                  for comp in predicted]  # 漏检目标

        for s in range(obs_len):
            tmp = 0
            qz = np.concatenate([qz_temp[:s], qz_temp[s+1:]])
            for k in range(1, obs_len+1):
                tmp = tmp + beta[k]*alpha[k]*delta(qz, k-1)
            tmp = tmp/pred_n
            for i, tmpgmm in enumerate(newobsgmm[s]):
                tmpgmm.weight = tmpgmm.weight * tmp/L_z
            newgmm.extend(newobsgmm[s])

        self.gmm = newgmm

        L_zn = np.zeros(self.N_max+1)
        for n in range(self.N_max+1):
            L_zn[n] = 0
            for j in range(min(obs_len, n)+1):
                L_zn[n] = L_zn[n] + beta[j]*factorial(n)/factorial(n-j)*np.power(1-self.detection, n-j)*delta(qz_temp, j)

        self.filter_pn = L_zn/L_z*pred_pn


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

    def extractstates(self):
        """
        选择权重大于的分量的状态
        :return:
        """
        items = []
        print("weights:")
        print([np.round(comp.weight, 7)
               for comp in self.gmm])
        number = sum(self.filter_pn[i]*i for i in range(self.N_max+1))
        num = np.round(number)
        if num ==0:
            return []
        else:
            for j, comp in enumerate(self.gmm):
                if j < num:
                    items.append(deepcopy(comp.loc))
        return items