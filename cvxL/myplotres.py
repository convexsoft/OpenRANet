import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


pd_cnncvxL = pd.read_csv('../cvxL_res_2by2/cnncvxl.csv',header=None, sep=' ')
print(pd_cnncvxL)
pd_cnncvxL[pd_cnncvxL<0]=0

pd_cnn = pd.read_csv('../cvxL_res_2by2/cnn_res1.csv',header=None, sep=' ')
print(len(pd_cnn.index))

index_num = len(pd_cnn.index)

random_forest = [0.08331278, 0.00034951, 0.00090057, 0.11674591]
marklist = [0,9,19,29,39,49,59,69,79,89,99]
color_choice = ['blue', 'green', 'orange', 'red']
alpha = 0.6


def plot_power():
    for i in range(4):
        plt.figure(figsize=(9, 7))
        print("i:",i)
        plt.plot(pd_cnncvxL.iloc[:,i], label="Algorithm 3",  linestyle = "-", linewidth=1, color = color_choice[0],markersize = 5, marker = "o",markevery = marklist, alpha=alpha)
        plt.plot(pd_cnn.iloc[:,i], linewidth=1, label="DNN", color = color_choice[1],marker = "^",markevery = marklist,alpha=alpha)
        plt.plot([random_forest[i]]*index_num, linewidth=1,  label="Random forest",color = color_choice[2],marker = "s",markersize = 4, markevery = marklist,alpha=alpha)
        plt.plot(pd_cnncvxL.iloc[:,i+4], linewidth=1, label="Ground truth",color = color_choice[3],marker = "x",markersize = 6, markevery = marklist,alpha=alpha)

        plt.legend(fontsize=24)
        plt.xlabel("epochs", fontsize=24)
        plt.ylabel("power(W)", fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        name = "../cvxL_res_2by2/power"+str(i)+".pdf"
        plt.savefig(name)
        # plt.show()

#cnn
G1 = np.array([[0.714,0.0315,0.084,0.7280000000000001]]).reshape(2, 2)
G2 = np.array([[0.6825000000000001,0.052000000000000005,0.0963,0.8255]]).reshape(2, 2)
sigma1 = np.array([[0.0505,0.05550000000000001]]).reshape(2, 1)
sigma2 = np.array([[0.057499999999999996,0.05500000000000001]]).reshape(2, 1)
r_bar = np.array([[1.1400000000000001,1.9]]).reshape(2, 1)
F1 = np.dot(np.linalg.pinv(np.diag(np.diag(G1))), G1 - np.diag(np.diag(G1)) )
F2 = np.dot(np.linalg.pinv(np.diag(np.diag(G2))), G2 - np.diag(np.diag(G2)) )
v1 = np.dot(np.linalg.pinv(np.diag(np.diag(G1))), sigma1)
v2 = np.dot(np.linalg.pinv(np.diag(np.diag(G2))), sigma2)
cnn_r_1 = []
cnn_r_2 = []

for j in range( len(pd_cnn.index)):
    pd_cnn_row = pd_cnn.iloc[j:j+1,0:4].values
    sinr1 = (1 / (np.dot(F1, pd_cnn_row[0][0:2].reshape(2,1)) + v1)) * pd_cnn_row[0][0:2].reshape(2,1)  # 2*1,m=1
    sinr2 = (1 / (np.dot(F2, pd_cnn_row[0][2:4].reshape(2,1)) + v2)) * pd_cnn_row[0][2:4].reshape(2,1)
    cnn_r_1.append(np.sum(np.sum(sinr1)))
    cnn_r_2.append(np.sum(np.sum(sinr2)))

#cvx
cvx_r_1 = [1.1400000000000001]*100
cvx_r_2 = [1.9]*100

random_forest_r_1 = (1 / (np.dot(F1, np.array([random_forest[0:2]]).reshape(2,1)) + v1)) * np.array([random_forest[0:2]]).reshape(2,1)  # 2*1,m=1
random_forest_r_2 = (1 / (np.dot(F2, np.array([random_forest[2:4]]).reshape(2,1)) + v2)) * np.array([random_forest[2:4]]).reshape(2,1)
random_forest_r_1 = [np.sum(np.sum(random_forest_r_1))]*100
random_forest_r_2 = [np.sum(np.sum(random_forest_r_2))]*100


def plot_rbar():
    color_choice = ['blue', 'green', 'orange', 'red']
    alpha = 0.6
    # r1_bar
    plt.figure(figsize=(9, 7))
    plt.plot(cvx_r_1, label="Algorithm 3", linestyle="-", linewidth=1, color=color_choice[0],
             markersize=5, marker="o", markevery=marklist, alpha=alpha)
    plt.plot(cnn_r_1, linewidth=1, label="DNN", color=color_choice[1], marker="^", markevery=marklist,
             alpha=alpha)
    plt.plot(random_forest_r_1, linewidth=1, label="Random forest", color=color_choice[2], marker="s",
             markersize=4, markevery=marklist, alpha=alpha)
    plt.plot(cvx_r_1, linewidth=1, label="Ground truth", color=color_choice[3], marker="x",
             markersize=6, markevery=marklist, alpha=alpha)

    plt.legend(fontsize=24)
    plt.xlabel("epochs", fontsize=24)
    plt.ylabel(r'$\bar{r}_1$', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    name = "../cvxL_res_2by2/r_1" + ".pdf"
    plt.savefig(name)
    plt.show()
    #r2_bar
    plt.figure(figsize=(9, 7))
    plt.plot(cvx_r_2, label="Algorithm 3", linestyle="-", linewidth=1, color=color_choice[0],
             markersize=5, marker="o", markevery=marklist, alpha=alpha)
    plt.plot(cnn_r_2, linewidth=1, label="DNN", color=color_choice[1], marker="^", markevery=marklist,
             alpha=alpha)
    plt.plot(random_forest_r_2, linewidth=1, label="Random forest", color=color_choice[2], marker="s",
             markersize=4, markevery=marklist, alpha=alpha)
    plt.plot(cvx_r_2, linewidth=1, label="Ground truth", color=color_choice[3], marker="x",
             markersize=6, markevery=marklist, alpha=alpha)

    plt.legend(fontsize=24)
    plt.xlabel("epochs", fontsize=24)
    plt.ylabel(r'$\bar{r}_2$', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    name = "../cvxL_res_2by2/r_2" + ".pdf"
    plt.savefig(name)
    plt.show()

if __name__ == '__main__':
    # plot_power()
    plot_rbar()
