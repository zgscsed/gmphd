#!/usr/bin/env python
#!-*-coding:utf-8 -*-
# Time    :2020/5/15 15:23
# Author  : zhoudong
# File    : gate_meas.py

"""
筛选量测数据
"""
import numpy as np
def gate_meas_kf(meas, gamma, prediced, model):
    """

    :param meas:
    :param gamma:
    :param prediced:
    :return:
    """
    valid_idx = []
    meas_len = len(meas)

    for comp in prediced:
        s = model.R + np.dot(np.dot(model.H, comp.cov), model.H.T)    # 协方差
        sqrt_sj = np.linalg.cholesky(s)


