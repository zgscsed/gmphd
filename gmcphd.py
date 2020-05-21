#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/14 12:58
# Author  : zhoudong
# File    : gmcphd.py

from copy import deepcopy
from model.cphd_model import *
import util.esf as esf

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
        self.cdn_update = np.zeros(self.N_max+1)      # 势分布
        self.cdn_update[1] = 1


    def update(self, obs):
        """
        预测 更新，
        :param obs: 当前时刻的观察值
        :return:
        """

        #############################################
        # step1 - prediction for birth targets
        birth = [deepcopy(comp) for comp in self.birthgmm]

        ##############################################
        # step2 - prediction for existing targets
        updated = [GmphdComponent(
            self.survival * comp.weight,
            np.dot(self.f, comp.loc),
            self.q + np.dot(np.dot(self.f, comp.cov), self.f.T)
        ) for comp in self.gmm]

        predicted = birth + updated

        # 更新势分布
        ############################################################################################
        # cardinalitu prediction
        survice_cdn_predict = np.zeros(self.N_max+1)                    # +1 有0, 1, ....20个
        for j in range(self.N_max+1):
            terms = np.zeros(self.N_max+1)
            for ell in range(j, self.N_max+1):
                terms[ell] = factorial(ell)/factorial(j)/factorial(ell-j)*\
                             np.power(self.survival, j)*np.power(1-self.survival, ell-j) * self.cdn_update[ell]
            survice_cdn_predict[j] = sum(terms)
        # predicted cardinality = convolution of birth and surviving cardinality distribution
        cdn_predict = np.zeros(self.N_max+1)                              # 预测的势分布
        for n in range(self.N_max+1):
            terms = np.zeros(self.N_max+1)
            for j in range(n+1):
                terms[j] = poisspdf(n-j, sum(comp.weight for comp in self.birthgmm)) * survice_cdn_predict[j]
            cdn_predict[n] = sum(terms)
        # 归一化
        cdn_predict = cdn_predict /sum(cdn_predict)

        #######################################################################################################
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
        qz_temp = []                         # 保存
        newobsgmm =[]
        for anobs in obs:
            anobs = np.array(anobs).reshape((2, 1))
            newgmmpartial = []
            qz_j = 0
            for j, comp in enumerate(predicted):
                weight_temp = self.detection*comp.weight*dmvnorm(nu[j], s[j], anobs)
                qz_j += weight_temp
                newgmmpartial.append(GmphdComponent(
                    weight_temp,
                    comp.loc + np.dot(k[j], anobs - nu[j]),
                    pkk[j]
                ))
            qz_temp.append(qz_j)
            newobsgmm.append(newgmmpartial)


        qz_temp = np.array(qz_temp)
        qz = qz_temp/self.model.pdf_c

        qz =qz.flatten()

        obs_len = len(obs)
        esfvaks_e = esf.esf(qz)          # 注意esfvaks_e 的size = qz+1
        esfvaks_d = np.zeros((obs_len, obs_len))
        for i in range(obs_len):
            esfvaks_d[i, :] = esf.esf(np.concatenate([qz[:i], qz[i + 1:]]))

        upsilon0_e = np.zeros(self.N_max+1)
        upsilon1_e = np.zeros(self.N_max+1)
        upsilon1_d = np.zeros((self.N_max+1, obs_len))

        for n in range(self.N_max+1):
            terms0_e = np.zeros(min(obs_len, n)+1)
            for j in range(min(obs_len, n)+1):
                terms0_e[j] = np.exp(-self.model.lambda_c+(obs_len-j)*np.log(self.model.lambda_c) +
                                     sum(np.log(i) for i in range(1, n)) - sum(np.log(i) for i in range(1, n-j))+
                                     (n-j)*np.log(1-self.model.P_D)-
                                     j*np.log(sum([comp.weight for comp in predicted])))*esfvaks_e[j]
            upsilon0_e[n] = sum(terms0_e)

            terms1_e = np.zeros(min(obs_len, n)+1)
            for j in range(min(obs_len, n)+1):
                if n >= j+1:
                    terms1_e[j] = np.exp(-self.model.lambda_c+(obs_len-j)*np.log(self.model.lambda_c)+
                                         sum(np.log(i) for i in range(1,n)) - sum(np.log(i) for i in range(1, n-j-1))+
                                         (n-j-1)*np.log(1-self.model.P_D)-
                                         (j+1)*np.log(sum([comp.weight for comp in predicted])))*esfvaks_e[j]
            upsilon1_e[n] = sum(terms1_e)

            if obs_len != 0:
                terms1_d = np.zeros((obs_len, min(obs_len-1, n)+1))
                for ell in range(obs_len):
                    for j in range(min(obs_len-1, n)+1):
                        # terms1_d[ell][j] = np.exp(-self.model.lambda_c+(obs_len-1-j)*np.log(self.model.lambda_c)+
                        #                         sum(np.log(i) for i in range(1, n)) - sum(np.log(i) for i in range(1, n-j-1))+
                        #                           (n-j-1)*np.log(1-self.model.P_D)-
                        #                           (j+1)*np.log(sum([comp.weight for comp in predicted])))*esfvaks_d[ell][j]
                        if n >= j+1:
                            terms1_d[ell][j] = np.exp(-self.model.lambda_c+(obs_len-j)*np.log(self.model.lambda_c)+
                                             sum(np.log(i) for i in range(1,n)) - sum(np.log(i) for i in range(1, n-j-1))+
                                             (n-j-1)*np.log(1-self.model.P_D)-
                                             (j+1)*np.log(sum([comp.weight for comp in predicted])))*esfvaks_d[ell][j]
                upsilon1_d[n, :] = np.sum(terms1_d, axis=1)

        # 漏检目标权值更新
        temp = np.dot(upsilon1_e, cdn_predict) / np.dot(upsilon0_e, cdn_predict)
        newgmm = [GmphdComponent(temp*comp.weight * (1.0 - self.detection), comp.loc, comp.cov)
                  for comp in predicted]  # 漏检目标

        for ell in range(obs_len):
            temp = np.dot(upsilon1_d[:, ell], cdn_predict) / np.dot(upsilon0_e, cdn_predict)
            for j, comp in enumerate(newobsgmm[ell]):
                comp.weight = comp.weight/self.model.pdf_c * temp
            newgmm.extend(newobsgmm[ell])


        self.gmm = newgmm

        # 更新
        self.cdn_update = upsilon0_e*cdn_predict
        self.cdn_update = self.cdn_update/sum(self.cdn_update)


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


