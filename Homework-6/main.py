# !/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Zhiquan Wang'
__date__ = '2018/11/18 17:08'

import MultipleLinearRegression as mlr
import numpy as np
from scipy.stats import f
from scipy.stats import t
if __name__ == "__main__":
    #test
    A = np.mat([[1,1,1],[1,1,1]])
    B = np.mat([[2,3,2],[2,2,2]])
    B[:,0] = A[:,0]+B[:,1]
    print(B)
    # Read Data
    tmp_model = mlr.MultipleLinearRegression()
    tmp_model.load_data(file_name='HW6Q2.txt')
    tmp_model.init_model()
    # H0 Beta1=Beta1=0
    h_0_result = tmp_model.hypothesis_test_zero(index_list=[1,2], alpha=0.05)
    print("Beta1 = Beta2 = 0",h_0_result)

    # H0 Beta1 = 1 Beta2 = 2
    sse_full = float(tmp_model.error_mat.T * tmp_model.error_mat)
    tmp_y_mat = tmp_model.y_mat - 1 * tmp_model.design_mat[:,1] - 2 * tmp_model.design_mat[:,2]
    tmp_design_mat = np.delete(tmp_model.design_mat, [1,2], axis=1)
    tmp_beta_mat = (tmp_design_mat.T * tmp_design_mat).I * tmp_design_mat.T * tmp_model.y_mat
    tmp_yr_hat_mat = tmp_design_mat * tmp_beta_mat
    tmp_error_mat = tmp_model.y_mat - tmp_yr_hat_mat
    sse_residual = tmp_error_mat.T * tmp_error_mat
    df_sse_f = tmp_model.sample_num - tmp_model.predictor_num
    df_sse_r = df_sse_f + 2
    tmp_value = float(((sse_residual - sse_full) / (df_sse_r - df_sse_f)) / (sse_full / df_sse_f))
    f_value = f.isf(0.01, df_sse_r - df_sse_f, df_sse_f)
    ht = tmp_value < f_value
    print("Beta1 = 2, Beta2 = 2",[ht, tmp_value, f_value])

    # H0 Beta2 = Beta3
    sse_full = float(tmp_model.error_mat.T * tmp_model.error_mat)
    tmp_design_mat = tmp_model.design_mat
    tmp_design_mat[:,2] = tmp_design_mat[:,2] + tmp_design_mat[:,3]
    tmp_design_mat = np.delete(tmp_model.design_mat, [3], axis=1)
    tmp_beta_mat = (tmp_design_mat.T * tmp_design_mat).I * tmp_design_mat.T * tmp_model.y_mat
    tmp_yr_hat_mat = tmp_design_mat * tmp_beta_mat
    tmp_error_mat = tmp_model.y_mat - tmp_yr_hat_mat
    sse_residual = tmp_error_mat.T * tmp_error_mat
    df_sse_f = tmp_model.sample_num - tmp_model.predictor_num
    df_sse_r = df_sse_f + 2
    tmp_value = float(((sse_residual - sse_full) / (df_sse_r - df_sse_f)) / (sse_full / df_sse_f))
    f_value = f.isf(0.01, df_sse_r - df_sse_f, df_sse_f)
    ht = tmp_value < f_value
    print("Beta1 = 2, Beta2 = 2", [ht, tmp_value, f_value])

