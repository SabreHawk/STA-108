# !/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Zhiquan Wang'
__date__ = '2018/10/18'

from scipy.stats import t
from scipy.stats import f
import math


class SimpleLinearModel(object):
    def __init__(self, *, x_data, y_data):
        self.__x_data = x_data
        self.__y_data = y_data
        self.__b1 = 0
        self.__b0 = 0
        self.__mean_x = 0
        self.__mean_y = 0
        self.__MSE = 0
        self.__expectation_sigma_hat_square = 0
        self.__estimator_variance_b1 = 0
        self.__estimator_variance_b0 = 0
        self.__std_x = 0
        self.__std_y = 0
        self.__sxx = 0
        self.__MSE = 0
        self.__estimator_variance_b1 = 0
        self.__estimator_variance_b0 = 0
        self.__regression_analysis()

    def __regression_analysis(self):
        self.__mean_x = sum(self.__x_data) / len(self.__x_data)
        self.__mean_y = sum(self.__y_data) / len(self.__y_data)
        self.__sxx = sum(
            [pow((xh - self.__mean_x), 2) for xh in self.__x_data])
        self.__b1 = sum([(self.__x_data[i] - self.__mean_x) *
                         (self.__y_data[i] - self.__mean_y)
                         for i in range(len(self.__x_data))]) / self.__sxx
        self.__b0 = self.__mean_y - self.__b1 * self.__mean_x
        self.__cal_MSE()
        self.__estimator_variance_b0_b1()

    @property
    def x_data(self):
        return self.__x_data

    @property
    def y_data(self):
        return self.__y_data

    @property
    def dataset(self):
        return [self.__x_data, self.__y_data]

    @dataset.setter
    def dataset(self, *, x_data, y_data):
        self.__x_data = x_data
        self.__y_data = y_data
        self.__regression_analysis()

    @property
    def b1(self):
        return self.__b1

    @property
    def b0(self):
        return self.__b0

    @property
    def mean_x(self):
        return self.__mean_x

    @property
    def mean_y(self):
        return self.__mean_y

    @property
    def MSE(self):
        return self.__MSE

    @property
    def sxx(self):
        return self.__sxx

    @property
    def estimator_variance_b0(self):
        return self.__estimator_variance_b0

    @property
    def estimator_variance_b1(self):
        return self.__estimator_variance_b1

    def y_hat(self, *, x):
        return self.__b0 + self.__b1 * x

    def __estimator_variance_b0_b1(self):
        self.__estimator_variance_b1 = self.__MSE / self.__sxx
        self.__estimator_variance_b0 = self.__MSE * (
            1 / len(self.__x_data) + pow(self.__mean_x, 2) / self.__sxx)

    def __cal_MSE(self):
        self.__MSE = sum([
            pow((self.__y_data[i] - self.y_hat(x=self.__x_data[i])), 2)
            for i in range(len(self.__y_data))
        ]) / (len(self.__y_data) - 2)

    def estimator_variance_yh(self, *, xh):
        return self.__MSE * (
            1 / len(self.__x_data) + pow(xh - self.__mean_x, 2) / self.__sxx)

    def confidence_interval_b0(self, *, alpha):
        tmp_t_value = t.isf(alpha * 2, len(self.__x_data) - 2)
        return [
            self.__b0 - tmp_t_value * math.sqrt(self.__estimator_variance_b0),
            self.__b0 + tmp_t_value * math.sqrt(self.__estimator_variance_b0)
        ]

    def confidence_interval_b1(self, *, alpha):
        tmp_t_value = t.isf(alpha / 2, len(self.__x_data) - 2)
        print(tmp_t_value)
        return [
            self.__b1 - tmp_t_value * math.sqrt(self.__estimator_variance_b1),
            self.__b1 + tmp_t_value * math.sqrt(self.__estimator_variance_b1)
        ]

    def confidence_interval_yh(self, *, xh, alpha):
        tmp_t_value = t.isf(alpha / 2, len(self.__x_data) - 2)
        print('2a - t value : ', tmp_t_value)
        print("2a - Y hat : ", self.y_hat(x=xh))
        print('2a - S^2 : ', self.estimator_variance_yh(xh=xh))
        return [
            self.y_hat(x=xh) -
            tmp_t_value * math.sqrt(self.estimator_variance_yh(xh=xh)),
            self.y_hat(x=xh) +
            tmp_t_value * math.sqrt(self.estimator_variance_yh(xh=xh))
        ]

    def multi_confidence_interval_yh(self, *, x_list, alpha):
        input_num = len(x_list)
        tmp_t_value = t.isf(alpha / (2 * input_num), len(self.__x_data) - 2)
        print('2b - t value : ', tmp_t_value)
        return_list = []
        for i in range(input_num):
            tmp_yh = self.y_hat(x=x_list[i])
            print('2b - CI Yh : ', tmp_yh)
            tmp_value = math.sqrt(self.estimator_variance_yh(xh=x_list[i]))
            print('2b - CI se_yh : ', tmp_value)
            return_list.append([
                tmp_yh - tmp_t_value * tmp_value,
                tmp_yh + tmp_t_value * tmp_value
            ])
        return return_list

    def multi_confidence_interval_yh_wh(self, *, x_list, alpha):
        input_num = len(x_list)
        tmp_f_value = math.sqrt(2 * f.isf(alpha, 2, len(self.__x_data) - 2))
        print('2c - f', tmp_f_value)
        return_list = []
        for i in range(input_num):
            tmp_yh = self.y_hat(x=x_list[i])
            print('Yh = ', tmp_yh)
            tmp_value = math.sqrt(self.estimator_variance_yh(xh=x_list[i]))
            print('tmp_v = ', tmp_value)

            return_list.append([
                tmp_yh - tmp_f_value * tmp_value,
                tmp_yh + tmp_f_value * tmp_value
            ])
        return return_list

    def prefiction_interval_yh(self, *, xh, alpha):
        tmp_t_value = t.isf(alpha / 2, len(self.__x_data) - 2)
        tmp_value = math.sqrt(self.__MSE * (1 + 1 / len(self.__x_data) + pow(
            (xh - self.__mean_x), 2) / self.__sxx))

        return [
            self.y_hat(x=xh) - tmp_t_value * tmp_value,
            self.y_hat(x=xh) + tmp_t_value * tmp_value
        ]

    def multi_preficition_interval_yh(self, *, x_list, alpha):
        input_num = len(x_list)
        tmp_t_value = t.isf(alpha / (2 * input_num), len(self.__x_data) - 2)
        return_list = []
        for i in range(input_num):
            tmp_yh = self.y_hat(x=x_list[i])
            print('2b - PI Yh : ', tmp_yh)
            tmp_value = math.sqrt(
                self.__MSE * (1 + 1 / len(self.__x_data) + pow(
                    (x_list[i] - self.__mean_x), 2) / self.__sxx))
            print('2b - PI tmp_value : ', tmp_value)
            return_list.append([
                tmp_yh - tmp_t_value * tmp_value,
                tmp_yh + tmp_t_value * tmp_value
            ])
        return return_list