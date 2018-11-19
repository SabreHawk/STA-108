# !/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Zhiquan Wang'
__date__ = '2018/10/18'

from scipy.stats import t
from scipy.stats import f
import math
import SimpleLinearModel as sl

with open('weight_full.txt', 'r') as f:
    x_list = []
    y_list = []
    for line in f:
        line = line.strip()
        [x, y] = line.split()
        x_list.append(float(x))
        y_list.append(float(y))

tmp_model = sl.SimpleLinearModel(x_data=x_list, y_data=y_list)
print('2a - b0', tmp_model.b0)
print('2a - b1', tmp_model.b1)
print('Sxx : ', tmp_model.sxx)
print('MSE : ', tmp_model.MSE)
print('95% C.I. : ', tmp_model.confidence_interval_yh(xh=175, alpha=0.05))
print('95% P.I. : ', tmp_model.prefiction_interval_yh(xh=175, alpha=0.05))
print(
    'Simu CI 174 175 179 95%',
    tmp_model.multi_confidence_interval_yh(x_list=[174, 175, 179], alpha=0.05))
print(
    'Simu PI 174 175 179 95%',
    tmp_model.multi_preficition_interval_yh(
        x_list=[174, 175, 179], alpha=0.05))
print(
    tmp_model.multi_confidence_interval_yh_wh(
        x_list=[174, 175, 179], alpha=0.05))
print(
    tmp_model.multi_confidence_interval_yh(
        x_list=[170, 172, 174, 175, 179, 182, 183, 186], alpha=0.05))
print(
    tmp_model.multi_confidence_interval_yh_wh(
        x_list=[170, 172, 174, 175, 179, 182, 183, 186], alpha=0.05))
