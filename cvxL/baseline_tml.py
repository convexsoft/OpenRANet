# tradictional machine learning

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
import csv
import numpy as np
import random

cvxcnn_data = list()
cvxcnn_target = list()
with open("../data_generation/data6.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        cvxcnn_data.append(list(map(float, line[:-4])))
        cvxcnn_target.append(list(map(float, line[-4:])))
        # print("cvxcnn_data:",cvxcnn_data)
X = np.array(cvxcnn_data)
y = np.array(cvxcnn_target)
# X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)
# print("x:",X)
# print("y:",y)
# idx = random.randint(0,200)
idx = 49
X_test = X[idx]
y_test = y[idx]
print("idx:",idx)
print("y_test:",y_test)


def forest(X, y, X_test):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    data_in = [X_test]
    yhat = model.predict(data_in)

    #,0.228940151251968,0.26662477276794255,0.2615691033823502,0.31073949824429486
    # ,0.21380560525860703,0.21022132204110144,0.4161908233431581,0.4628740922712697
    print (yhat)


def forest_obj(X, y, X_test):
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)

    data_in = [X_test]
    yhat = model.predict(data_in)

    #,0.228940151251968,0.26662477276794255,0.2615691033823502,0.31073949824429486
    # ,0.21380560525860703,0.21022132204110144,0.4161908233431581,0.4628740922712697
    return np.sum(np.sum(yhat))


def linearsvr(X, y, X_test):
    from sklearn.datasets import make_regression
    from sklearn.multioutput import RegressorChain
    from sklearn.svm import LinearSVR
    # create datasets
    # define model
    model = LinearSVR()
    wrapper = RegressorChain(model)
    # fit model
    wrapper.fit(X, y)
    # make a prediction
    data_in = [X_test]
    yhat = wrapper.predict(data_in)
    # summarize prediction
    print(yhat[0])


if __name__ == '__main__':
    # run_time = 10
    # res_list = []
    # for i in range(run_time):
    #     res = forest_obj(X, y, X_test)
    #     res_list.append(res)
    # print(res_list)
    linearsvr(X, y, X_test)
    forest(X, y, X_test)
