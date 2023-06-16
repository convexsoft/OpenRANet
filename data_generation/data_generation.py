import numpy as np


def obj_func(p1,p2):
    obj = np.sum(p1)+np.sum(p2)
    return obj

def constarint_func(p1,p2):
    G1 = np.array([[0.8, 0.12], [0.08, 0.85]])  # m=1
    G2 = np.array([[0.85, 0.10], [0.09, 0.85]])  # m=2
    sigma1 = np.array([[0.05, 0.05]])
    sigma2 = np.array([[0.05, 0.05]])
    r_bar = np.array([[1.5,2.5]]).T


    F1 = np.dot(np.linalg.inv(np.diag(np.diag(G1))), (G1 - np.diag(np.diag(G1))))
    F2 = np.dot(np.linalg.inv(np.diag(np.diag(G2))), (G2 - np.diag(np.diag(G2))))
    v1 = np.dot(np.linalg.inv(np.diag(np.diag(G1))), sigma1.T)
    v2 = np.dot(np.linalg.inv(np.diag(np.diag(G2))), sigma2.T)

    sinr1 = (1 / (np.dot(F1, p1.T) + v1)) * p1.T  # 2*1,m=1
    sinr2 = (1 / (np.dot(F2, p2.T) + v2)) * p2.T  # 2*1,m=2

    f1_func = np.log(1+sinr1)
    f2_func = np.log(1+sinr2)

    # print(f1_func)
    # print(f1_func+f2_func)
    if (f1_func+f2_func>=r_bar).all():
        return True
    else:
        return False


if __name__ == '__main__':
    p1 = np.array([[0.05,0.2]])
    p2 = np.array([[0.1,0.3]])

    diff = 0.01
    max_p = 5
    obj_min = 10000

    for p11 in np.arange(0.001,max_p,diff):
        print("p11:", p11)
        for p12 in np.arange(0.001, max_p, diff):
            for p21 in np.arange(0.001, max_p, diff):
                for p22 in np.arange(0.001, max_p, diff):
                    p1 = np.array([[p11, p12]])
                    p2 = np.array([[p21,p22]])
                    # print("p1:", p1)
                    # print("p2:", p2)
                    res = constarint_func(p1,p2)
                    # print(res)
                    if res == True:
                        obj = obj_func(p1,p2)
                        if obj <= obj_min:
                            obj_min = obj

    print("p1:", p1)
    print("p2:", p2)
    print("obj_min:", obj_min)



