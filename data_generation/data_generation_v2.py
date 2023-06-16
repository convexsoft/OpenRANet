import numpy as np
import random as rd
import pandas as pd


def obj_func(p1,p2):
    obj = np.sum(p1)+np.sum(p2)
    return obj


def interation_p(r_bar, r_11,r_21, G1, G2, sigma1, sigma2):
    p1 = np.array([[0.05, 0.2]]).T
    p2 = np.array([[0.1, 0.3]]).T

    r_12 = r_bar[0][0] - r_11
    r_22 = r_bar[0][1] - r_21
    r1 = np.array([[r_11, r_12]]).T
    r2 = np.array([[r_21, r_22]]).T

    F1 = np.dot(np.linalg.inv(np.diag(np.diag(G1))), (G1 - np.diag(np.diag(G1))))
    F2 = np.dot(np.linalg.inv(np.diag(np.diag(G2))), (G2 - np.diag(np.diag(G2))))
    v1 = np.dot(np.linalg.inv(np.diag(np.diag(G1))), sigma1)
    v2 = np.dot(np.linalg.inv(np.diag(np.diag(G2))), sigma2)

    epsilon = 0.0001
    err_p = 1
    while err_p >= epsilon:
        sinr1 = (1 / (np.dot(F1, p1) + v1)) * p1  # 2*1,m=1
        sinr2 = (1 / (np.dot(F2, p2) + v2)) * p2  # 2*1,m=2
        f1_func = np.log(1 + sinr1)
        f2_func = np.log(1 + sinr2)
        p1_temp = p1
        p2_temp = p2
        p1 = r1 * p1 / f1_func
        p2 = r2 * p2 / f2_func
        err_p = np.linalg.norm(p1-p1_temp, ord = 2) + np.linalg.norm(p2-p2_temp, ord = 2)

    err_x = 1
    x1 = np.array([[0.05, 0.2]]).T
    x2 = np.array([[0.1, 0.3]]).T

    sinr1 = (1 / (np.dot(F1, p1) + v1)) * p1  # 2*1,m=1
    sinr2 = (1 / (np.dot(F2, p2) + v2)) * p2  # 2*1,m=2

    while err_x >= epsilon:
        x1 = 1 + np.dot(F1, sinr1 * x1)
        x2 = 1 + np.dot(F2, sinr2 * x2)
        x1_temp = x1
        x2_temp = x2
        err_x = np.linalg.norm(x1 - x1_temp, ord=2) + np.linalg.norm(x2 - x2_temp, ord=2)
    lambda_1 = r1*p1*x1*(1/(1-(1/np.exp(r1))))
    lambda_2 = r2*p2*x2*(1/(1-(1/np.exp(r2))))

    return p1,p2,lambda_1,lambda_2


def generate_param():
    r_bar_std = np.array([[1.5, 2.5]])
    G1_std = np.array([[0.8, 0.03], [0.08, 0.85]])  # m=1
    G2_std = np.array([[0.85, 0.05], [0.09, 0.85]])  # m=2
    sigma1_std = np.array([[0.05, 0.05]]).T
    sigma2_std = np.array([[0.05, 0.05]]).T

    r_bar = r_bar_std * np.array([[1+round(rd.random(),2), 1+round(rd.random(),2)]])
    G1 = G1_std * np.array([[1+round(0.1*rd.random(),2), 1+round(0.1*rd.random(),2)],[1+round(0.1*rd.random(),2), 1+round(0.1*rd.random(),2)]])  # m=1
    G2 = G2_std * np.array([[1+round(0.1*rd.random(),2), 1+round(0.1*rd.random(),2)],[1+round(0.1*rd.random(),2), 1+round(0.1*rd.random(),2)]])  # m=2
    sigma1 = sigma1_std * np.array([[1+round(0.1*rd.random(),2), 1+round(0.1*rd.random(),2)]]).T
    sigma2 = sigma2_std * np.array([[1+round(0.1*rd.random(),2), 1+round(0.1*rd.random(),2)]]).T

    print("G1, G2, sigma1, sigma2:",G1, G2, sigma1, sigma2)
    return r_bar, G1, G2, sigma1, sigma2


if __name__ == '__main__':

    obj_min = 1000
    diff = 0.01
    p1_opt = 0
    p2_opt = 0
    lambda_1_opt = 0
    lambda_2_opt = 0
    r_11_opt = 0
    r_21_opt = 0

    num = 1
    for i in range(num):

        r_bar, G1, G2, sigma1, sigma2 = generate_param()

        for r_11 in np.arange(0.01, r_bar[0][0], diff):
            print("r_11:", r_11)

            for r_21 in np.arange(0.01, r_bar[0][1], diff):
                p1,p2,lambda_1,lambda_2 = interation_p(r_bar, r_11,r_21, G1, G2, sigma1, sigma2)
                obj = obj_func(p1,p2)
                if obj <= obj_min:
                    obj_min = obj
                    p1_opt = p1
                    p2_opt = p2
                    lambda_1_opt = lambda_1
                    lambda_2_opt = lambda_2
                    r_11_opt = r_11
                    r_21_opt = r_21

        print("p1_opt:", p1_opt)
        print("p2_opt:", p2_opt)
        print("lambda_1_opt:", lambda_1_opt)
        print("lambda_2_opt:", lambda_2_opt)
        print("r_11_opt:", r_11_opt)
        print("r_21_opt:", r_21_opt)
        print("obj_min:", obj_min)


