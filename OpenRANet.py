import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import random
import scipy
import csv

torch.random.seed()
def data_progress():
    cvxcnn_data = list()
    cvxcnn_target = list()
    with open("../data_generation/data6.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            cvxcnn_data.append(list(map(float, line[:-4])))
            cvxcnn_target.append(list(map(float, line[-4:])))
            # print("cvxcnn_data:",cvxcnn_data)

    cvxcnn_data = np.array(cvxcnn_data)
    cvxcnn_target = np.array(cvxcnn_target)
    x_train,x_test,y_train,y_test = train_test_split(cvxcnn_data,cvxcnn_target,test_size=0.001,random_state=42)
    print("x_train:",x_train.shape,"x_test:",x_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape) #(14448, 8) (6192, 8) (14448,) (6192,)

    # scale = StandardScaler()
    # x_train = scale.fit_transform(x_train)
    # x_test = scale.transform(x_test)
    train_xt = torch.from_numpy(x_train.astype(np.float32))
    train_yt = torch.from_numpy(y_train.astype(np.float32))
    test_xt = torch.from_numpy(x_test.astype(np.float32))
    test_yt = torch.from_numpy(y_test.astype(np.float32))

    print("train_xt:", train_xt)
    print("train_yt:", train_yt)

    train_data = Data.TensorDataset(train_xt,train_yt)

    test_data = Data.TensorDataset(test_xt,test_yt)
    train_loader = Data.DataLoader(dataset=train_data,batch_size=1,shuffle=True,num_workers=0)
    print("train_loader:", train_loader)

    idx = random.randint(0,299)
    #idx = 249，132，272，160
    idx = 49
    print("idx:",idx)
    single_data = np.array([cvxcnn_data[idx].tolist()])
    single_target = cvxcnn_target[idx]
    # single_data = scale.transform(single_data)
    single_data = torch.from_numpy(single_data.astype(np.float32))
    return train_loader, train_xt, train_yt, test_xt, test_yt,y_test,single_data,single_target


class Cvxnnregression(nn.Module):
    def __init__(self):
        super(Cvxnnregression, self).__init__()
        self.neuron_num = 100
        self.hidden1 = nn.Linear(in_features=16,out_features=self.neuron_num,bias=True) #200
        self.hidden2 = nn.Linear(self.neuron_num,self.neuron_num) #200*200
        self.hidden3 = nn.Linear(self.neuron_num,self.neuron_num) #200*100
        self.r_layer = nn.Linear(self.neuron_num,4)

    def forward(self,x):
        x0 = x
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        r_predict = self.r_layer(x)
        r_project = self.projectionL(r_predict, x0)
        p_predict = self.cvxL( r_project,x0)
        p_predict_f = F.relu( p_predict)

        return p_predict_f, r_predict

    def projectionL(self, r, x0):
        r_bar = torch.reshape(x0[0][12:14], (2,1))
        L = 2
        diag_r_bar = []
        for v in r_bar:
            diag_r_bar.extend([v[0]] * L)
        diag_r_bar = torch.diag(torch.tensor(diag_r_bar))

        one_m = np.ones((L, L))
        block_diag = one_m
        for i in range(len(r_bar) - 1):
            block_diag = scipy.linalg.block_diag(block_diag, one_m)
        block_diag = torch.from_numpy(block_diag.astype(np.float32))
        r_tilde = torch.div(torch.mm(diag_r_bar, r.reshape(4,1)), torch.mm(block_diag, r.reshape(4,1)))
        # r_tilde =  torch.mm(torch.linalg.pinv(torch.diag(torch.mm(block_diag, r.reshape(4,1)).t()[0]) ), torch.mm(diag_r_bar, r.reshape(4,1)))

        return r_tilde.t()

    def cvxL(self, predict_r,x0):
        G1 = x0[0][0:4].reshape(2, 2)
        G2 = x0[0][4:8].reshape(2, 2)

        sigma1 = x0[0][8:10].reshape(2, 1)
        sigma2 = x0[0][10:12].reshape(2, 1)

        F1 = torch.mm(torch.linalg.pinv(torch.diag(torch.diag(G1))), G1 - torch.diag(torch.diag(G1)) )
        F2 = torch.mm(torch.linalg.pinv(torch.diag(torch.diag(G2))), G2 - torch.diag(torch.diag(G2)) )
        v1 = torch.mm(torch.linalg.pinv(torch.diag(torch.diag(G1))), sigma1)
        v2 = torch.mm(torch.linalg.pinv(torch.diag(torch.diag(G2))), sigma2)

        # p1_star = torch.mm(torch.linalg.pinv(torch.eye(2)-torch.mm(torch.diag(torch.exp(predict_r[0][0:2])-1), F1)),v1).t()
        # p2_star = torch.mm(torch.linalg.pinv(torch.eye(2)-torch.mm(torch.diag(torch.exp(predict_r[0][2:4])-1), F2)),v2).t()

        p1_star = torch.mm(torch.mm(torch.linalg.inv(torch.eye(2) - torch.mm(torch.diag(predict_r[0][0:2]), F1)), torch.diag(predict_r[0][0:2])), v1)
        p2_star = torch.mm(torch.mm(torch.linalg.inv(torch.eye(2) - torch.mm(torch.diag(predict_r[0][2:4]), F2)), torch.diag(predict_r[0][2:4])), v2)

        res = torch.cat((p1_star,p2_star), 0).t()
        return res




def custom_mse(predicted, target):
    total_mse = 0

    # target = torch.tensor([[0.,0.,0.,0.]])  # add this line for unsupervised learning
    for i in range(target.shape[1]):
        # print("predicted[i]:", predicted.T[i])
        total_mse+=nn.MSELoss()(predicted.T[i], target.T[i])
    return total_mse


def custom_mse_v2(p_predict, r_predict, b_y, b_r):  # p_predict, r_predict, b_y, b_r)
    total_mse = 0
    for i in range(b_y.shape[1]):
        # print("predicted[i]:", predicted.T[i])
        total_mse+=nn.MSELoss()(p_predict.T[i], b_y.T[i]) + nn.MSELoss()(r_predict.T[i], b_r.T[i])
    return total_mse



def training_progress(train_loader, train_xt, train_yt, test_xt, test_yt,y_test,single_data,single_target):
    epoch_num = 100
    cvxcnnreg = Cvxnnregression()
    optimizer = SGD(cvxcnnreg.parameters(),lr=0.0015,weight_decay=0.0001)
    loss_func = nn.MSELoss() 
    train_loss_all = []
    single_tp = []
    optimal_value = single_target.tolist()
    optimal_value = [optimal_value]*epoch_num
    # print("optimal_value:", optimal_value)

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        train_loss = 0
        train_num = 0
        for step,(b_x,b_y) in enumerate(train_loader):
            # print("step:", step)
            p_predict, r_predict = cvxcnnreg(b_x)

            # print("output:", output)
            # loss = loss_func(output,b_y)

            r_bar = b_x[0][12:14]
            r_opy_1 = b_x[0][14:16]
            r_opy_2 = r_bar - r_opy_1
            b_r = torch.tensor([[r_opy_1[0], r_opy_2[0],r_opy_1[1], r_opy_2[1]]])
            loss = custom_mse(p_predict, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        single_predict, r_predict = cvxcnnreg(single_data)
        p  = single_predict.data.numpy()
        # print("====p:",p)
        # single_tp.append(np.sum(p))
        single_tp.append(p[0].tolist())
        print("train_loss_all:", train_loss / train_num)
    print("train_loss_all:",train_loss_all)
    return cvxcnnreg, single_tp,optimal_value

def plot_power(single_tp,optimal_value):
    single_tp = np.array(single_tp)
    # print("single_tp:", single_tp)
    optimal_value = np.array(optimal_value)
    # print("optimal_value:", optimal_value)

    with open('../cvxL_res_2by2/cnncvxl_relu2.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quoting=csv.QUOTE_MINIMAL)

        for i in range(single_tp.shape[0]):
            add_row = []
            add_row.extend(single_tp[i])
            add_row.extend(optimal_value[i])
            spamwriter.writerow(add_row)

    # plot loss
    plt.figure(figsize=(8,6))
    color_choice =  ['red','blue','green','purple']

    for i in range(4):
        print("i:",i)
        plt.plot(single_tp[:,i], label="Algorithm 3", color = color_choice[i], alpha=0.7)
        plt.plot(optimal_value[:,i], linewidth=2, linestyle="-." , label="Ground truth",color = color_choice[i],alpha=0.7)
    # plt.legend(fontsize=14)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("power(W)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("../cvxL_res_2by2/cnncvxl_relu2.pdf")
    plt.show()


if __name__ == '__main__':
    run_time = 1
    obj_list = []
    train_loader, train_xt, train_yt, test_xt, test_yt, y_test, single_data, single_target = data_progress()
    if run_time == 1:
        cvxcnnreg, single_tp,optimal_value = training_progress(train_loader, train_xt, train_yt, test_xt, test_yt, y_test,single_data,single_target)
        plot_power(single_tp, optimal_value)
    if run_time > 1:
        for i in range(run_time):
            cvxcnnreg, single_tp, optimal_value = training_progress(train_loader, train_xt, train_yt, test_xt, test_yt,
                                                                    y_test, single_data, single_target)
            single_predict, r_predict = cvxcnnreg(single_data)
            p = single_predict.data.numpy()
            obj_list.append(np.sum(p[0]))

        print(obj_list)
        add_row = []
        with open('../cvxL_res_2by2/obj_list_cnn.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quoting=csv.QUOTE_MINIMAL)

            spamwriter.writerow(obj_list)



