# !/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Zhiquan Wang'
__date__ = '2018/11/5 15:25'

import copy
import math
import numpy as np
from scipy.stats import f
from scipy.stats import t


class MultipleLinearRegression(object):
    def __init__(self):
        self.__sample_num = 0
        self.__predictor_num = 0
        self.__x_mat = None
        self.__design_mat = None
        self.__y_mat = None
        self.__beta_hat_mat = None
        self.__hat_mat = None
        self.__y_hat_mat = None
        self.__error_mat = None
        self.__sigma_square_hat = None
        self.__variance_beta_hat = None

    def load_data(self, *, file_name):
        tmp_data = np.loadtxt(file_name)
        self.__y_mat = np.mat(tmp_data[:, 0]).T
        self.__x_mat = np.mat(tmp_data[:, 1:])
        self.__sample_num = len(self.__y_mat)
        self.__design_mat = np.hstack((np.ones((self.__sample_num, 1)), self.__x_mat))
        self.__predictor_num = self.__design_mat.shape[1]

    def init_model(self):
        self.__beta_hat_mat = (self.__design_mat.T * self.__design_mat).I * self.__design_mat.T * self.__y_mat
        self.__hat_mat = self.__design_mat * (self.__design_mat.T * self.__design_mat).I * self.__design_mat.T
        self.__y_hat_mat = self.__hat_mat * self.__y_mat
        self.__error_mat = self.__y_mat - self.__y_hat_mat
        self.__sigma_square_hat = float((self.__error_mat.T * self.__error_mat) / (
                self.__sample_num - self.__predictor_num))
        self.__variance_beta_hat = self.__sigma_square_hat * (self.__design_mat.T * self.__design_mat).I

    def variance_y_hat(self, *, x_list):
        x_hat_mat = np.mat(x_list).T
        return self.__sigma_square_hat * x_hat_mat.T * (self.__design_mat.T * self.__design_mat).I * x_hat_mat

    def hypothesis_test_zero(self, *, index_list, alpha):
        sse_full = float(self.__error_mat.T * self.__error_mat)
        print("SSE_Full = ", sse_full)
        tmp_design_mat = np.delete(self.__design_mat, index_list, axis=1)
        tmp_beta_mat = (tmp_design_mat.T * tmp_design_mat).I * tmp_design_mat.T * self.__y_mat
        tmp_yr_hat_mat = tmp_design_mat * tmp_beta_mat
        tmp_error_mat = self.__y_mat - tmp_yr_hat_mat
        sse_residual = tmp_error_mat.T * tmp_error_mat
        print("SSE_Residual = ", sse_residual)
        df_sse_f = self.__sample_num - self.__predictor_num
        # print("DF_full = ",df_sse_f)
        df_sse_r = df_sse_f + len(index_list)
        print("DF_residual = ", df_sse_r)
        tmp_value = float(((sse_residual - sse_full) / (df_sse_r - df_sse_f)) / (sse_full / df_sse_f))
        f_value = f.isf(alpha, df_sse_r - df_sse_f, df_sse_f)

        ht = tmp_value < f_value
        return [ht, tmp_value, f_value]

    def confidence_interval_y_hat(self, *, x_list, alpha):
        tmp_x_mat = np.mat(x_list).T
        pre_num = tmp_x_mat.shape[1]
        tmp_y_hat = tmp_x_mat.T * self.__beta_hat_mat
        print('Y_Hat = ', tmp_y_hat)
        t_value = t.isf(alpha / (2 * pre_num), self.__sample_num - self.__predictor_num)
        tmp_sd = np.sqrt(self.variance_y_hat(x_list=x_list))
        return [np.diag(tmp_y_hat - t_value * tmp_sd), np.diag(tmp_y_hat + t_value * tmp_sd)]

    def prediction_interval_y_hat(self, *, x_list, alpha):
        tmp_x_mat = np.mat(x_list).T
        pre_num = tmp_x_mat.shape[1]
        tmp_y_hat = tmp_x_mat.T * self.__beta_hat_mat
        t_value = t.isf(alpha / (2 * pre_num), self.__sample_num - self.__predictor_num)
        tmp_sd = np.sqrt(self.__sigma_square_hat * (
                1 + tmp_x_mat.T * (self.__design_mat.T * self.__design_mat).I * tmp_x_mat))
        return np.mat([np.diag(tmp_y_hat - t_value * tmp_sd), np.diag(tmp_y_hat + t_value * tmp_sd)]).T

    @property
    def sample_num(self):
        return self.__sample_num

    @property
    def predictor_num(self):
        return self.__predictor_num

    @property
    def x_mat(self):
        return self.__x_mat

    @property
    def y_mat(self):
        return self.__y_mat

    @property
    def design_mat(self):
        return self.__design_mat

    @property
    def beta_hat_mat(self):
        return self.__beta_hat_mat

    @property
    def hat_mat(self):
        return self.__hat_mat

    @property
    def y_hat_mat(self):
        return self.__y_hat_mat

    @property
    def error_mat(self):
        return self.__error_mat

    @property
    def sigma_square_hat(self):
        return self.__sigma_square_hat

    @property
    def variance_beta_hat(self):
        return self.__variance_beta_hat
