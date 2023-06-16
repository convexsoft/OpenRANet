import torch
import torch.nn as nn

# Set  seed
torch.manual_seed(2)
import sklearn
from sklearn import datasets
from sklearn import preprocessing
iris = datasets.load_iris()
X = torch.tensor(preprocessing.normalize(iris.data[:, :2]), dtype=torch.float)
y = torch.tensor(iris.target.reshape(-1, 1), dtype=torch.float)

print(X.size())
print(y.size())

X = X[:y[y < 2].size()[0]]
y = y[:y[y < 2].size()[0]]
print(X.size())
print(y.size())

X, y = sklearn.datasets.make_moons(200, noise=0.20)
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
print(X.size())
print(y.size())


class FNN(nn.Module):
    def __init__(self, ):
        super().__init__()

        # Dimensions for input, hidden and output
        self.input_dim = 2
        self.hidden_dim = 32
        self.output_dim = 1

        # Learning rate definition
        self.learning_rate = 0.05

        # Our parameters (weights)
        self.w1 = torch.randn(self.input_dim, self.hidden_dim)

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
        y4 = self.sigmoid(self.y3)
        return y4

    # Backward propagation
    def backward(self, X, l, y4):
        self.dC_dy4 = y4 - l

        '''
        Gradients for w2: partial derivative of cost w.r.t. w2
        dC/dw2
        '''
        self.dy4_dy3 = self.sigmoid_first_order_derivative(y4)
        self.dy3_dw2 = self.y2

        # Y4 delta: dC_dy4 dy4_dy3
        self.y4_delta = self.dC_dy4 * self.dy4_dy3

        self.dC_dw2 = torch.matmul(torch.t(self.dy3_dw2), self.y4_delta)

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

    def train(self, X, l):
        # Forward propagation
        y4 = self.forward(X)

        # Backward propagation and gradient descent
        self.backward(X, l, y4)


model = FNN()

loss_lst = []

num_epochs = 201

for epoch in range(num_epochs):
    # Get our predictions
    y_hat = model(X)

    cross_entropy_loss = -(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))
    mean_cross_entropy_loss = torch.mean(cross_entropy_loss).detach().item()
    if epoch % 20 == 0:
        print('Epoch {} | Loss: {}'.format(epoch, mean_cross_entropy_loss))
    loss_lst.append(mean_cross_entropy_loss)

    model.train(X, y)








