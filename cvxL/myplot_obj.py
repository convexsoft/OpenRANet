import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
#sns.set(style="darkgrid")
#sns.set(style="whitegrid")
#sns.set_style("white")
sns.set(style="whitegrid",font_scale=2)
import matplotlib.collections as clt
import ptitprince as pt
import matplotlib as mpl
import seaborn as sns


savefigs = True
# sns.set(color_codes=True)
# mpl.rcParams["axes.unicode_minus"] = False

def export_fig(axis,text, fname):
    if savefigs:
        axis.text()
        axis.savefig(fname, bbox_inches='tight')


# Changing orientation

def all_cloud():
    data = {'Equal rates': [0.1734485, 0.1734485, 0.1734485, 0.1734485, 0.1734485, 0.1734485, 0.1734485, 0.1734485, 0.1734485, 0.1734485],
          'Algorithm 2': [0.20266476, 0.17046141, 0.18521613, 0.1804485, 0.20988892, 0.19944905, 0.19469398, 0.18907965, 0.18324354, 0.17944569],
          'Algorithm 3': [0.20966476, 0.21046141, 0.21521613, 0.2104485, 0.20988892, 0.20944905, 0.21469398, 0.21407965, 0.21324354, 0.21044569],
          'DNN': [0.18630211, 0.18220109, 0.18859944, 0.18049143, 0.19394238, 0.17894998, 0.18567711, 0.19673717, 0.1801385, 0.18032846],
          'Random forest': [0.20491863609337554, 0.2032325252945379, 0.20485777386113602, 0.20873503315370429, 0.2067409834400139, 0.20519913937086592, 0.2050135298352508, 0.2030573036281402, 0.20405182154434398, 0.20666873882609377]
          }
    df = pd.DataFrame(data)
    df.plot.box(grid=False)
    plt.xticks(rotation = -5, fontsize=13)
    plt.yticks(fontsize=12)
    plt.ylabel("Optimal value", fontsize=13)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    name = "../cvxL_res_2by2/obj" + ".pdf"
    plt.savefig(name)
    plt.show()

def plot_iterations():
    alg = ['Equal rates','Algorithm 2','Algorithm 3','DNN','Random forest']
    value = [18,86,6,4,1]
    plt.bar(alg,value, 0.4)
    plt.grid(False)
    plt.ylabel("Layers", fontsize=13)
    plt.xticks(rotation = -5, fontsize=13)
    plt.yticks(fontsize=12)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    name = "../cvxL_res_2by2/layers" + ".pdf"
    plt.savefig(name)
    plt.show()


if __name__ == '__main__':
    all_cloud()
    # plot_iterations()




