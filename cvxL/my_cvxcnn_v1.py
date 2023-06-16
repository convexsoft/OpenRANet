import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import csv
import random
torch.manual_seed(2)
import scipy.linalg
from torch.autograd.functional import jacobian


def data_progress():
    cvxcnn_data = list()
    cvxcnn_target = list()
    with open("../data_generation/data3.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            cvxcnn_data.append(list(map(float, line[:-4])))
            cvxcnn_target.append(list(map(float, line[-4:])))

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

    train_data = Data.TensorDataset(train_xt,train_yt)

    test_data = Data.TensorDataset(test_xt,test_yt)
    train_loader = Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=0)
    idx = random.randint(0,299)
    #idx = 249，132，272，160
    idx = 149
    print("idx:",idx)
    single_data = np.array([cvxcnn_data[idx].tolist()])
    single_target = cvxcnn_target[idx]
    # single_data = scale.transform(single_data)
    single_data = torch.from_numpy(single_data.astype(np.float32))
    data_dict = {"train_loader":train_loader, "train_xt":train_xt, "train_yt":train_yt, "test_xt":test_xt,
                 "test_yt":test_yt,"y_test":y_test,"single_data":single_data,"single_target":single_target}
    return data_dict


class FNN(nn.Module):
    def __init__(self, ):
        super().__init__()

        # Dimensions for input, hidden and output
        self.input_dim = 16
        self.hidden_dim = 32
        self.output_dim = 4

        # Learning rate definition
        self.learning_rate = 0.02

        # Our parameters (weights)
        # w1: 2 x 32
        self.w1 = torch.randn(self.input_dim, self.hidden_dim)

        # w2: 32 x 1
        self.w2 = torch.randn(self.hidden_dim, self.output_dim)

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoid_first_order_derivative(self, s):
        return s * (1 - s)

    # Forward propagation
    def forward(self, X):
        # First linear layer
        self.y1 = torch.matmul(X, self.w1) # 3 X 3 ".dot" does not broadcast in PyTorch
        # First non-linearity
        self.y2 = self.sigmoid(self.y1)
        # Second linear layer
        self.y3 = torch.matmul(self.y2, self.w2)
        # Second non-linearity
        self.y4 = self.sigmoid(self.y3)

        #===cvxl
        ycvx = []
        # r_tilde = []
        for i,x in enumerate(X):

            #projection layer
            r = self.y4[i]
            r_tilde_sgl = self.projection_v2(r, x)
            # r_tilde.append(r_tilde_sgl)
            #cvx layer
            G1 = x[0:4].numpy().reshape(2,2)
            G2 = x[4:8].numpy().reshape(2,2)
            sigma1 = x[8:10].numpy().reshape(2,1)
            sigma2 = x[10:12].numpy().reshape(2,1)
            # r_bar = x[12:14].numpy().reshape(2,1)

            # print(G1, G2, sigma1, sigma2, r_bar, r_11, r_21, )
            r_tilde_sgl = r_tilde_sgl.numpy().reshape(4,1)
            p1, p2 = self.interation_p(r_tilde_sgl, G1, G2, sigma1, sigma2)
            p = np.concatenate([p1.reshape(1,2),p2.reshape(1,2)],axis=1)
            ycvx.append(p)
        ycvx = np.concatenate(ycvx,axis=0)
        ycvx = torch.from_numpy(ycvx.astype(np.float32))
        # r_tilde = torch.cat(r_tilde, dim=1).t()
        return ycvx  #, r_tilde

    # def projection(self,r,r_bar):
    #     r1_tilde = r[0:2]*r_bar[0]/np.sum(r[0:2])
    #     r2_tilde = r[2:4]*r_bar[1]/np.sum(r[2:4])
    #     r_tilde = np.concatenate([r1_tilde,r2_tilde],axis=0)
    #     return r_tilde

    def projection_v2(self, r, x):
        r_bar = torch.reshape(x[12:14], (2,1))
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
        r_tilde = torch.mm(diag_r_bar, r.reshape(4,1)) / torch.mm(block_diag, r.reshape(4,1))
        return r_tilde.reshape(4)


    def interation_p(self, r_tilde, G1, G2, sigma1, sigma2):
        p1 = np.array([[0.05, 0.2]]).T
        p2 = np.array([[0.1, 0.3]]).T

        # r_12 = r_bar[0][0] - r_11
        # r_22 = r_bar[0][1] - r_21
        # r1 = np.array([[r_11, r_12]]).T
        # r2 = np.array([[r_21, r_22]]).T
        r1 = r_tilde[0:2].reshape(2,1)
        r2 = r_tilde[2:4].reshape(2,1)

        F1 = np.dot(np.linalg.inv(np.diag(np.diag(G1))), (G1 - np.diag(np.diag(G1))))
        F2 = np.dot(np.linalg.inv(np.diag(np.diag(G2))), (G2 - np.diag(np.diag(G2))))
        v1 = np.dot(np.linalg.inv(np.diag(np.diag(G1))), sigma1)
        v2 = np.dot(np.linalg.inv(np.diag(np.diag(G2))), sigma2)

        epsilon = 0.0001
        err_p = 1
        try:
            while err_p >= epsilon:
                sinr1 = (1 / (np.dot(F1, p1) + v1)) * p1  # 2*1,m=1
                sinr2 = (1 / (np.dot(F2, p2) + v2)) * p2  # 2*1,m=2
                f1_func = np.log(1 + sinr1)
                f2_func = np.log(1 + sinr2)
                p1_temp = p1
                p2_temp = p2
                p1 = r1 * p1 / f1_func
                p2 = r2 * p2 / f2_func

                err_p = np.linalg.norm(p1 - p1_temp, ord=2) + np.linalg.norm(p2 - p2_temp, ord=2)

        except Exception:
            print("infeasible problem!")
            return 0, 0


        return p1, p2

    # Backward propagation
    def backward(self, X, Y, ycvx):

        # Derivative of binary cross entropy cost w.r.t. final output y4
        self.dC_dy4 = ycvx - Y
        # self.y4 = self.dC_dy4
        '''
        Gradients for r(w): partial derivative of cost w.r.t. r(w): 
        dL/dr_tilde(w) = 2*(Phi(r(w))-p) * d Phi(r_tilde(w))/ dr_tilde(w) => 2*(ycvx-y) * dycvx/ dr_tilde
        '''
        self.dL_dycvx = 2*(ycvx - Y)


        def vec(x, y):
            print("torch.cat([x,y]):", torch.cat([a.view(-1) for b in [x, y] for a in b]))
            return torch.cat([a.view(-1) for b in [x, y] for a in b])

        dycvx_dr = []
        for i,x in enumerate(X):
            p = torch.reshape(ycvx[i], (4,1))
            lamb = p
            w = vec(p, lamb)
            J_cvx = jacobian(lambda z: vec(*self.kkt(z[0:4], z[4:8], x)), w)

            kkt_dr_tilde = torch.cat((torch.zeros(len(p),len(p)),torch.diag(lamb.t()[0])),0)
            dycvx_lamb_dr_tilde = torch.matmul(torch.linalg.inv(J_cvx), kkt_dr_tilde)
            # print(dycvx_lamb_dr)
            dycvx_dr_tilde_sgl = dycvx_lamb_dr_tilde[0:len(p)]
            # print("dycvx_dr:", dycvx_dr_tilde)

            # print(dycvx_lamb_dr_tilde)
            r = self.y4[i]
            J_project_sgl = jacobian(lambda z: self.projection_v2(z, x), r)
            print("J_project_sgl:", J_project_sgl)
            dycvx_dr_sgl = torch.mm(dycvx_dr_tilde_sgl,J_project_sgl)
            dycvx_dr.append(dycvx_dr_sgl)

        dycvx_dr = torch.stack(dycvx_dr,0)
        print("dycvx_dr:", dycvx_dr)
        '''
        Gradients for w2: partial derivative of cost w.r.t. w2
        dC/dw2
        '''
        self.dy4_dy3 = self.sigmoid_first_order_derivative(self.y4)
        self.dy3_dw2 = self.y2

        # Y4 delta: dC_dy4 dy4_dy3
        self.y4_delta = self.dC_dy4 * self.dy4_dy3

        # This is our gradients for w1: dC_dy4 dy4_dy3 dy3_dw2
        self.dC_dw2 = torch.matmul(torch.t(self.dy3_dw2), self.y4_delta)

        self.dC_dw2 = torch.matmul(dycvx_dr,self.dC_dw2)
        '''
        Gradients for w1: partial derivative of cost w.r.t w1
        dC/dw1
        '''
        self.dy3_dy2 = self.w2
        self.dy2_dy1 = self.sigmoid_first_order_derivative(self.y2)

        # Y2 delta: (dC_dy4 dy4_dy3) dy3_dy2 dy2_dy1
        self.y2_delta = torch.matmul(self.y4_delta, torch.t(self.dy3_dy2)) * self.dy2_dy1

        # Gradients for w1: (dC_dy4 dy4_dy3) dy3_dy2 dy2_dy1 dy1_dw1
        self.dC_dw1 = torch.matmul(torch.t(X), self.y2_delta)

        # Gradient descent on the weights from our 2 linear layers
        self.w1 -= self.learning_rate * self.dC_dw1
        self.w2 -= self.learning_rate * self.dC_dw2


    def projection_dir(self, r_tilde, r_bar):
        dr_bar_dr_tilde1 = r_tilde[0:2]*r_bar[0]/np.sum(r_tilde[0:2])
        dr_bar_dr_tilde2 = r_tilde[2:4]*r_bar[1]/np.sum(r_tilde[2:4])
        return np.concatenate([dr_bar_dr_tilde1,dr_bar_dr_tilde2],axis=0)


    def kkt(self, p, lamb, x):
        G1 = torch.reshape(x[0:4], (2, 2))
        G2 = torch.reshape(x[4:8], (2, 2))
        sigma1 = torch.reshape(x[8:10], (2, 1))
        sigma2 = torch.reshape(x[10:12], (2, 1))
        # r_bar = torch.reshape(x[12:14], (1, 2))
        r_1 = torch.reshape(x[14:16], (2, 1))
        r_2 = torch.reshape(x[14:16], (2, 1))

        p = torch.reshape(p,(4,1))
        lamb = torch.reshape(lamb,(4,1))

        p.requires_grad_()
        lamb.requires_grad_()
        obj_func = torch.sum(torch.sum(torch.exp(p)))
        F1 = torch.mm(torch.linalg.inv(torch.diag(torch.diag(G1))), (G1 - torch.diag(torch.diag(G1))))
        F2 = torch.mm(torch.linalg.inv(torch.diag(torch.diag(G2))), (G2 - torch.diag(torch.diag(G2))))
        v1 = torch.mm(torch.linalg.inv(torch.diag(torch.diag(G1))), sigma1)
        v2 = torch.mm(torch.linalg.inv(torch.diag(torch.diag(G2))), sigma2)

        p1 = p[0:2]
        p2 = p[2:4]

        sinr1 = (1 / (torch.mm(F1, p1) + v1)) * p1  # 2*1,m=1
        sinr2 = (1 / (torch.mm(F2, p2) + v2)) * p2  # 2*1,m=2
        constraint1 = -torch.log(torch.log(1 + sinr1)) + torch.log(r_1)
        constraint2 = -torch.log(torch.log(1 + sinr2)) + torch.log(r_2)
        con = torch.cat((constraint1, constraint2), 0)
        # print("lamb:", lamb)

        lagarian = obj_func + lamb.t().mm(con)

        grad_lagarian = torch.autograd.grad(lagarian, p, create_graph=True)
        comp_slack = torch.mul(lamb, con)
        print("grad_lagarian:", grad_lagarian)
        return grad_lagarian[0], comp_slack


    def train_model(self, X, y):
        # Forward propagation
        ycvx = self.forward(X)

        # Backward propagation and gradient descent
        self.backward(X, y, ycvx)


def custom_mse(predicted, target):
    total_mse = 0
    # print("predicted:", predicted)
    # print("target:", target)

    for i in range(target.shape[1]):
        # print("predicted[i]:", predicted.T[i])
        total_mse+=nn.MSELoss()(predicted.T[i], target.T[i])
    return total_mse


def training():
    # Instantiate our model class and assign it to our model object
    model = FNN()

    # Loss list for plotting of loss behaviour
    loss_lst = []

    # Number of times we want our FNN to look at all 100 samples we have, 100 implies looking through 100x
    num_epochs = 10

    # data_dict = {"train_loader": train_loader, "train_xt": train_xt, "train_yt": train_yt, "test_xt": test_xt,
    #              "test_yt": test_yt, "y_test": y_test, "single_data": single_data, "single_target": single_target}
    data_dict = data_progress()
    X = data_dict["train_xt"]
    y = data_dict["train_yt"]

    # Let's train our model with 100 epochs
    for epoch in range(num_epochs):
        # Get our predictions
        y_hat = model(X)
        # print("X:", X[0:4])
        #
        # print("y_hat:", y_hat[0:4])

        # Cross entropy loss, remember this can never be negative by nature of the equation
        # But it does not mean the loss can't be negative for other loss functions
        # cross_entropy_loss = -(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))
        m_loss = custom_mse(y_hat, y)

        # We have to take cross entropy loss over all our samples, 100 in this 2-class iris dataset
        mean_m_loss = torch.mean(m_loss).detach().item()

        # Print our mean cross entropy loss
        if epoch % 20 == 0:
            print('Epoch {} | Loss: {}'.format(epoch, mean_m_loss))
        loss_lst.append(mean_m_loss)

        # (1) Forward propagation: to get our predictions to pass to our cross entropy loss function
        # (2) Back propagation: get our partial derivatives w.r.t. parameters (gradients)
        # (3) Gradient Descent: update our weights with our gradients
        model.train_model(X, y)


if __name__ == '__main__':
    training()

