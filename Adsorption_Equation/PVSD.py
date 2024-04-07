# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 03:22:55 2024

@author: JChonpca_Huang
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.optimize as opt

# FA

# [0.16722678, 0.16442365, 0.05086458, 0.16542762, 0.05074939, 0.13561252, 0.0976876 , 0.1707491 ]

# [0.03352138, 0.07404045, 0.05101445, 0.06060472, 0.04333584, 0.01758684, 0.03256742, 0.07389156]

# [0.16637952, 0.92536901, 0.0805736 , 0.49922464, 0.21616394, 0.0898183 , 0.61580155, 0.58438589]

# [0.41403644, 0.51241312, 0.07583296, 0.89605222, 0.2117562 , 0.7098126 , 0.18924946, 0.56843274]

# [0.10243063, 0.12034347, 0.08146878, 0.1490835 , 0.10936842, 0.04015449, 0.16495753, 0.03003341]

# [0.31869627, 6.81279379, 0.40771035, 0.32298684, 0.0760788 , 0.95405152, 0.0408935 , 0.09731724]








L = 0.15

g = 15

data = np.array(pd.read_excel('data.xlsx'))

data_t = data[:,0]/60


data_1 = data[:,9]

data_2 =  data[:,10]

data_1_t = data_t[~np.isnan(data_1)]

data_2_t = data_t[~np.isnan(data_2)]

data_1 = data_1[~np.isnan(data_1)]

data_2 = data_2[~np.isnan(data_2)]


data_1_q = -(data_1 - data_1[0])*L/g

data_2_q = -(data_2 - data_2[0])*L/g

data_q = np.hstack([data_1_q,data_2_q[1::]+data_1_q[-1]])

data_c = np.hstack([data_1[0:-1],data_2])


# data_t = data_t[0:8]

# data_c = data_c[0:8]


delta_t = 0.01

simulate_t = np.arange(0, data_t[-1] + delta_t, delta_t)

r = 1.0

delta_r = 0.01

simulate_r = np.arange(0, r + delta_t, delta_r)

data_array = np.zeros([simulate_t.shape[0],simulate_r.shape[0]+1])


location = []

for i in data_t:
    
    location.append(abs((simulate_t-i)).tolist().index(abs((simulate_t-i)).min()))


def goal(xx):
    
    print(xx)
    

    # parames definition
        
    
    # Li_r
    
    q_m = xx[0]
    
    K_a = xx[1]
    
    
    # C_r
    
    m = g
    
    v = L
    
    s = xx[2]
    
    k_f =  xx[3] #KL
    
    # Car_r
    
    e_p = xx[4] # fai*e
    
    p_a = xx[5]
    
    D_p = xx[6]
    
    D_s = xx[7]

    
    
    
    # init_data
    
    data_array[0,-1] = data_c[0]
    
    
    
    # run
    
    def pvsd(data_array_t, CA , q_m, K_a, m, v, s, k_f, e_p, p_a, D_p, D_s):
        
        
        
        def f(x):
            
            # global cpr_t_1
            # global cpr_t
            # global cpr_t_s_1
            # global cpr_t_s__1
            # global r_
            # global return_list
            
            
            x = x.reshape([-1,])
            
            return_list = [(x[0]-x[1])/delta_r, -k_f*(CA-x[-1])+(D_p+(D_s*p_a*q_m*K_a)/((1+K_a*x[-1])**2))*((x[-1]-x[-2])/delta_r)]
            
                        
            cpr_t_1 = x[1:-1]
            
            cpr_t = data_array_t[1:-2]
            
            
            cpr_t_s_1 = x[2::]
            
            cpr_t_s = cpr_t_1
            
            cpr_t_s__1 = x[0:-2]
            
            r_ = simulate_r[1:-1]

            
            diff_t = (e_p+(p_a*q_m*K_a)/((1+K_a*cpr_t_1)**2))*((cpr_t_1-cpr_t)/delta_t)
            
            eq_right = (D_p+(D_s*p_a*q_m*K_a)/((1+K_a*cpr_t_1)**2))*((cpr_t_s_1-2*cpr_t_s+cpr_t_s__1)/(delta_r**2)) - ((2*D_s*p_a*q_m*(K_a**2))/((1+K_a*cpr_t_1)**3))*(((cpr_t_s_1-cpr_t_s)/(delta_r))**2) + (2/r_)*(D_p+(D_s*p_a*q_m*K_a)/((1+K_a*cpr_t_1)**2))*((cpr_t_s_1-cpr_t_s)/delta_r)

            
            # eq_right = np.where(eq_right<0, 0, eq_right).reshape([-1,])
            
            un_bound = diff_t - eq_right
            
            un_bound = un_bound.tolist()
            
            un_bound.append(return_list[0])
            
            un_bound.append(return_list[1])
            
            return un_bound
            
        result = opt.fsolve(f, data_array_t[0:-1])
                
        return result
    
        
    for i in range(1,simulate_t.shape[0]):
        
        if i != 200:
            
        
            data_array_t = data_array[i-1]
            
            CA = max( data_array[i-1,-1] - max(m*s*k_f*(data_array[i-1,-1]-data_array[i-1,-2])*delta_t/v, 0), 0)
            
            data_array[i,-1] = CA
            
            data_array[i,0:-1] = pvsd(data_array_t, CA , q_m, K_a, m, v, s, k_f, e_p, p_a, D_p, D_s)
        
        
        else:
            
            data_array_t = data_array[i-1]
            
            CA = max( data_array[i-1,-1] - max(m*s*k_f*(data_array[i-1,-1]-data_array[i-1,-2])*delta_t/v, 0), 0) + 8
                        
            data_array[i,-1] = CA
            
            data_array[i,0:-1] = pvsd(data_array_t, CA , q_m, K_a, m, v, s, k_f, e_p, p_a, D_p, D_s)
        
        
    
    plt.plot(simulate_t, data_array[:,-1])
    
    plt.scatter(data_t, data_c)
    
    loss = ((data_array[:,-1][location].reshape(-1,)-data_c)**2).mean()
    
    plt.title(str(loss))
    
    plt.show()
    
    print(loss)

    return loss





lb_array = [5,0,0,0,0,0,0,0]

ub_array = [10,1,2,1,10,5,5,10]

lb_array = [0]*8

ub_array = [1,8] + [1]*6


# from sko.GA import GA

# ga = GA(func=goal, n_dim=8, size_pop=20, max_iter=20, prob_mut=0.1, lb=lb_array, ub=ub_array, precision=1e-100)

# import time

# a = time.time()

# best_x, best_y = ga.run()

# b = time.time()

# print('best_x:', best_x, '\n', 'best_y:', best_y)

# print('time spend:', b-a, 's')

# plt.plot(simulate_t,data_array[:,-1])

# plt.scatter(data_t, data_c)

# plt.xlabel('Time (h)')

# plt.ylabel('C (g/L)')

# plt.title('BA')

# plt.savefig('PVSD-BA.png',dpi=300)

# plt.show()








