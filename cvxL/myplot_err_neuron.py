import pandas as pd

import matplotlib.pyplot as plt


pd_neuron_num = pd.read_csv('../cvxL_res_2by2/err_neuron_num.csv',header=None, sep=',')
print(pd_neuron_num)

color_choice = ['blue', 'green', 'red','orange','purple']
label_list = ["50 neurons per layer", "75 neurons per layer", "100 neurons per layer", "150 neurons per layer", "200 neurons per layer"]
marker_list = ["o","^","v","x","s"]
alpha = 0.6

plt.figure(figsize=(9, 7))


def plot_err_neuron():
    for i in range(5):
        print("i:",i)
        plt.plot(pd_neuron_num.iloc[i,:], label=label_list[i],  linestyle = "-", linewidth=1, color = color_choice[i],markersize = 5, marker = marker_list[i],alpha=alpha)

    plt.legend(fontsize=21)
    plt.xlabel("epochs", fontsize=22)
    plt.ylabel("Mean square error", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=18)
    name = "../cvxL_res_2by2/err_neuron_num.pdf"
    plt.savefig(name)
    plt.show()

pd_learning_rate = pd.read_csv('../cvxL_res_2by2/err_learning_rate.csv',header=None, sep=',')
print(pd_learning_rate)
label_list2 = ["Learning rate: 0.0001","Learning rate: 0.0005","Learning rate: 0.001","Learning rate: 0.01","Learning rate: 0.1"]
def plot_learning_rate():
    for i in range(5):
        print("i:",i)
        plt.plot(pd_learning_rate.iloc[i,:], label=label_list2[i],  linestyle = "-", linewidth=1, color = color_choice[i],markersize = 5, marker = marker_list[i],alpha=alpha)

    plt.legend(fontsize=21)
    plt.xlabel("epochs", fontsize=22)
    plt.ylabel("Mean square error", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=18)
    name = "../cvxL_res_2by2/pd_learning_rate.pdf"
    plt.savefig(name)
    plt.show()


if __name__ == '__main__':
    plot_err_neuron()
    # plot_learning_rate()