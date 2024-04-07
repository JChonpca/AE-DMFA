# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 03:22:55 2024

@author: JChonpca_Huang
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize as opt


# FA

# [0.10674069, 0.0499184 , 0.03988327]

# [0.04382148, 0.03425188, 0.04536707]


# [0.1171202,0.06389531,0.04157714]

# [0.2191703,0.05289842,0.0225588 ]


# [0.11104639 0.06780233 0.01503909] 

# [0.35574218,0.10472578,0.03578375]



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

data_1_interp_func = interp1d(data_1_t, data_1, bounds_error=False, fill_value=(data_1[0], data_1[1]))

data_2_interp_func = interp1d(data_2_t, data_2, bounds_error=False, fill_value=(data_2[0], data_2[1]))

data_1_q = -(data_1 - data_1[0])*L/g

data_2_q = -(data_2 - data_2[0])*L/g

data_q = np.hstack([data_1_q,data_2_q[1::]+data_1_q[-1]])



delta_t = 0.01

simulate_t = np.arange(0, data_t[-1] + delta_t, delta_t)

data_array = np.zeros_like(simulate_t)

def Langmuir(tt,q,ka,qe,kd):
    
    
    def f(x):
        
        if x < 0:
            
            x = 0
        
        if tt < 2:
            
            return (x-q)/delta_t - ka*data_1_interp_func(tt)*(qe-q)-kd*q
        
        else:
            
            return (x-q)/delta_t - ka*data_2_interp_func(tt)*(qe-q)-kd*q
        
    result = opt.fsolve(f, q)
            
    return result


location = []

for i in data_t:
    
    location.append(abs((simulate_t-i)).tolist().index(abs((simulate_t-i)).min()))

def goal(x):
    
    global simulate_t
    global location
    
    ka = x[0]
    
    qe = x[1]
    
    kd = x[2]
    
    data_array[0] = data_q[0]
    
    for i in range(1,simulate_t.shape[0]):
        
        data_array[i] = Langmuir(simulate_t[i-1], data_array[i-1], ka, qe, kd)[0]
        
    
    loss = ((data_array[location].reshape(-1,)-data_q)**2).mean()
    
    return loss



# lb_array = [0,0,0]

# ub_array = [1,1,1]

# from sko.GA import GA

# ga = GA(func=goal, n_dim=3, size_pop=50, max_iter=50, prob_mut=0.1, lb=lb_array, ub=ub_array, precision=1e-100)

# import time

# a = time.time()

# best_x, best_y = ga.run()

# b = time.time()

# print('best_x:', best_x, '\n', 'best_y:', best_y)

# print('time spend:', b-a, 's')

# plt.plot(simulate_t,data_array)

# plt.scatter(data_t, data_q)

# plt.xlabel('Time (h)')

# plt.ylabel('q (mg/g)')

# plt.title('BA')

# # plt.savefig('L-BA.png',dpi=300)

# plt.show()


