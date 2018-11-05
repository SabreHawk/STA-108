# !/usr/bin/env python3
# -*- coding:utf-8 -*-
__author__ = 'Zhiquan Wang'
__date__ = '2018/10/30 16:09'

import numpy as np
import matplotlib.pyplot as plt
import SimpleLinearModel as slm

if __name__ == '__main__':
    # Problem a
    dataset_num = 7
    dataset = []
    plt.figure(figsize=(6, 3))
    for i in range(dataset_num):
        # read data set
        tmp_data = np.loadtxt('data%d.txt' % (i + 1))
        dataset.append(tmp_data)
        # cal r^2

        tmp_model = slm.SimpleLinearModel(x_data=tmp_data[:, 0], y_data=tmp_data[:, 1])
        print('R_square ',i+1," = ",tmp_model.r_square)


        # plot sub figure
        plt.subplot(2, 4, i + 1)

        y_hat_list = [tmp_model.y_hat(x=tmp_data[i][0]) for i in range(len(tmp_data[:, 0]))]
        plt.plot(tmp_data[:, 0], y_hat_list, color='red')
        plt.scatter(tmp_data[:, 0], tmp_data[:, 1], s=2)
        plt.ylim(-5, 30)
    # plot
    plt.show()

    # Problem b
    tmp_data = np.loadtxt('weight_full.txt')
    x_mat = np.hstack((np.ones((len(tmp_data[:, 0]), 1)), np.mat(tmp_data[:, 0]).T))
    y_mat = np.mat(tmp_data[:, 1]).T
    print("First five rows of x : \n", x_mat[0:5])
    print("First five rows of y : \n", y_mat[0:5])
    print("X^t*X : \n", x_mat.T * x_mat)
    print("X^t*y : \n", x_mat.T * y_mat)
    beta_hat = (x_mat.T * x_mat).I * (x_mat.T * y_mat)
    print("Beta Hat :\n", beta_hat)
    y_hat_mat = np.mat(beta_hat[0].mean() * x_mat[:, 0] + beta_hat[1].mean() * x_mat[:, 1])
    theta_square_hat = 0
    for i in range(len(y_hat_mat)):
        theta_square_hat += pow((y_mat[i] - y_hat_mat[i]), 2)
    theta_square_hat /= (len(tmp_data[:, 0]) - 2)
    print("theta_square_hat : ", theta_square_hat.mean())
    variance_beta_mat = theta_square_hat.mean() * (x_mat.T * x_mat).I
    print("Variance Beta Hat :\n", variance_beta_mat)
    tmp_model = slm.SimpleLinearModel(x_data=tmp_data[:, 0], y_data=tmp_data[:, 1])
    print("VB0 ", tmp_model.estimator_variance_b0)
    print("VB1 ", tmp_model.estimator_variance_b1)
    print("MSE ", tmp_model.MSE)
    print("B0 ", tmp_model.b0)
    print("B1 ", tmp_model.b1)
