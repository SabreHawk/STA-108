# !/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Zhiquan Wang'
__date__ = '2018/11/5 15:24'

import MultipleLinearRegression as mlr
import numpy as np
import math
if __name__ == '__main__':
    tmp_mode = mlr.MultipleLinearRegression()
    tmp_mode.load_data(file_name='property.txt')
    tmp_mode.init_model()
    np.set_printoptions(precision=10, suppress=True)

    print(tmp_mode.beta_hat_mat[2]/math.sqrt(tmp_mode.sigma_square_hat*(tmp_mode.design_mat.T*tmp_mode.design_mat).I[2,2]));
    print(tmp_mode.beta_hat_mat)
    print('The dimension of the design matrix X =', tmp_mode.design_mat.shape)
    print('The dimension of the response matrix Y =', tmp_mode.y_mat.shape)
    print("X^t * X =\n", tmp_mode.design_mat.T * tmp_mode.design_mat)
    print("X^t * Y =\n", tmp_mode.design_mat.T * tmp_mode.y_mat)
    print("(X^t * X)^-1 =\n", (tmp_mode.design_mat.T * tmp_mode.design_mat).I)
    print("Beta hat =\n", tmp_mode.beta_hat_mat)
    print("The first 6 cases of fitted values = \n", tmp_mode.y_hat_mat[0:6])
    print("The first 6 cases of residuals = \n", tmp_mode.error_mat[0:6])
    print("Estimate of sigma square =\n", np.mat(tmp_mode.sigma_square_hat))
    # print("Variance of beta hat =\n",tmp_mode.variance_beta_hat)
    print('Hypothesis test beta2 =0,signifiacne level = 0.01\n',
          tmp_mode.hypothesis_test_zero(index_list=[2], alpha=0.01))
    print('CI for x=[1,4,10,10,8],alpha = 0.05\n',
          tmp_mode.confidence_interval_y_hat(x_list=[[1, 4, 10, 10, 8]], alpha=0.05))
    print('PI for x=[1,4,10,10,8],alpha = 0.05\n',
          tmp_mode.prediction_interval_y_hat(x_list=[[1, 4, 10, 10, 8]], alpha=0.05))
    print('PI for x=[1,4,10,10,8],[1,5,11,8,12],alpha = 0.05\n',
          tmp_mode.prediction_interval_y_hat(x_list=[[1, 4, 10, 10, 8], [1, 5, 11, 8, 12]], alpha=0.05))
