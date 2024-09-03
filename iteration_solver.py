import numpy as np


def param_gen():
    m = 2
    L = 2

    G = np.zeros((m, L, L))
    G1 = np.array([[2,0.1],[0.2,1.2]])
    G2 = np.array([[2.9,0.1],[0.2,2.2]])
    G[0,:,:] = G1
    G[1,:,:] = G2
    r_bar = 10*np.array([[0.3,0.3],[0.2,0.2]])
    sigma = 0.001*np.array([[0.05,0.05],[0.05,0.05]])
    return G, sigma, r_bar


def sinr_func(G, p, sigma):
    dim = np.shape(p)
    sinr = np.zeros(dim)
    dim = list(dim)
    m = dim[0]
    for i in range(m):
        Gi = G[i,:,:]
        Gi_diag = np.diag(Gi)
        Gi_ndiag = Gi - np.diag(Gi_diag)
        pi = p[i,:]
        sigmai = sigma[i,:]
        sinri = np.dot(Gi_diag,pi)/(np.dot(Gi_ndiag,pi.T)+sigmai)
        sinr[i,:] = sinri
    # print("asd:", sinr)
    return sinr


def data_rate_func(sinr):
    dr_res = np.log(1+sinr)
    return dr_res


def data_rate_func_deriv(sinr):
    return 1/(1+sinr)


def iteration_alg_sum():
    k = 0
    t = 0
    p_iter = np.array([[0.5,0.4],[0.5,0.5]])
    x_iter = np.array([[0.5, 0.5], [0.5, 0.5]])
    epsilon = 10e-3
    dim = np.shape(p_iter)
    m = list(dim)[0]
    G, sigma, r_bar = param_gen()

    sinr = sinr_func(G, p_iter, sigma)
    dr_res = data_rate_func(sinr)
    print("dr_func:",dr_res)
    dr_col_sum = np.sum(dr_res, axis=0)
    dr_col_sum_inv = 1/dr_col_sum
    print("dr_col_sum:",dr_col_sum)
    print("dr_col_sum_inv:",dr_col_sum_inv)
    w_iter = np.dot(np.diag(dr_col_sum_inv), dr_res)
    print("w_iter:",w_iter)
    r_bar_l = np.sum(r_bar, axis=0)
    print("r_bar:",r_bar)
    print("r_bar_l:",r_bar_l)
    print("p_iter:",p_iter)
    p_iter = np.dot(np.diag(r_bar_l), p_iter)/(np.dot(np.diag(r_bar_l), (1-w_iter))+dr_res)
    print("p_iter:",p_iter)

    ## update x_iter
    # 1/f_{ll}^m
    deriv_dr_res = 1/(1+sinr)
    # F_m_ll =


def iteration_alg():
    p_iter = np.array([[0.5,0.4],[0.5,0.5]])
    x_iter = np.array([[0.5, 0.5], [0.5, 0.5]])
    err = 1
    dim = np.shape(p_iter)
    M = list(dim)[0]
    L = list(dim)[1]
    G, sigma, r_bar = param_gen()

    while err >= 10e-4:

        sinr = sinr_func(G, p_iter, sigma)
        dr_res = data_rate_func(sinr)
        # print("dr_func:",dr_res)
        dr_deriv = data_rate_func_deriv(sinr)
        p_temp = p_iter

        # print("p_iter:",p_iter)
        p_iter = (r_bar * p_iter) / dr_res

        # for m in range(M):
        #     for l in range(L):
        #         temp_ml = 0
        #         for h in range(L):
        #             if (h!=l):
        #                 # temp_ml = temp_ml + x_iter[m][h]*(1/dr_res[m][h])*dr_deriv[m][h]*(sinr[m][h])^2*G[m][h][l]/G[m][h][h]
        #                 temp_ml = temp_ml + x_iter[m,h]*(1/dr_res[m,h])*dr_deriv[m,h]*(sinr[m,h])**2*G[m,h,l]/G[m,h,h]
        #         x_iter[m,l] = dr_res[m,l]/(dr_deriv[m,l]*sinr[m,l]) + temp_ml

        lambda_iter = p_iter*x_iter
        print("p_iter:",p_iter)
        print("dr_func:",dr_res)
        print("r_bar:", r_bar)

        # err = np.linalg.norm(p_iter-p_temp, ord=1)
        err = np.linalg.norm(dr_res-r_bar, ord=1)

        print("err:",err)




if __name__ == '__main__':
    iteration_alg()




