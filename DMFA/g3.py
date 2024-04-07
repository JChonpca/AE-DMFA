# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:55:49 2022

@author: 577
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from scipy.interpolate import interp1d,barycentric_interpolate

models_for_adsorption = []

p1_for_adsorption = []


models = []

trans = []


seed = 577

np.random.seed(seed)

random.seed(seed)

mpl.rc('axes', lw=2)

data_ = np.array(pd.read_excel('./exp_oh/g3.xlsx',header=None))

data__ = np.array(pd.read_excel('./exp_gly/g3.xlsx',header=None))

data1 = data_[:-1,:].astype('float')

data2 = data__[:-1,:].astype('float')

data1[:,-1] /= 1000

data2[:,-1] /= 1000

acid_labels = ['Base','Feed Medium','F','V','D']

acid_color = ['blue','red','green','blue','red']

marker_style = ['^','o','s','o','^','s','v','*','>']

acid_top = [3.5,4,3,10]

acid_low = [2.5,3,2,9]


# plt.figure(dpi=300)

plt.rcParams['xtick.direction'] = 'in'

plt.rcParams['ytick.direction'] = 'in'


fig,ax = plt.subplots(1,2,figsize=(35,14),dpi=300)





ax1 = ax[0]

ax2 = ax1.twinx()

ax1.set_title('a',fontproperties = 'Arial',size=50,loc='left', x = -0.15)

ax1.errorbar(data1[:,0],data1[:,1],
              yerr=0,
              c=acid_color[0],
              label=acid_labels[0],
              marker=marker_style[0],
              markersize = 10,
              linestyle='None')


from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression,LassoLarsCV

from sklearn.preprocessing import PolynomialFeatures

p1 = PolynomialFeatures(degree=7)

p1.fit(data1[:,0:1].reshape(-1,1))

x_h = p1.transform(data1[:,0:1])

# model = LassoLarsCV(cv=5)

model = LinearRegression()

model.fit(x_h,data1[:,1])

models.append(model)

trans.append(p1)

print(model.coef_)

print(r2_score(data1[:,1],model.predict(x_h)))

xx = np.linspace(0, data1[-1,0], 50).reshape(-1,1)

x_h = p1.transform(xx)

ax1.plot(xx,np.where(model.predict(x_h)<0,0,model.predict(x_h)),c=acid_color[0],linewidth=7)

models_for_adsorption.append(model)

p1_for_adsorption.append(p1)



# from scipy.optimize import curve_fit

# def sigmoid(x, a, b, c):

#     y = c / (1 + np.exp(-b*(x-a)))

#     return y


# popt, pcov = curve_fit(sigmoid, data1[:,0],data1[:,1],bounds=([0,0,0], [20,10,50]))

# print(popt)

# print(r2_score(data1[:,1],sigmoid(data1[:,0], *popt)))

# xx = np.linspace(0, data1[-1,0], 50)

# yy = sigmoid(xx, *popt)

# print(popt)

# models.append(popt)

# trans.append('fuck')

# ax1.plot(xx,yy,c=acid_color[0],linewidth=3)



ax1.errorbar(data2[:,0],data2[:,1],
              yerr=0,
              c=acid_color[1],
              label=acid_labels[1],
              marker=marker_style[1],
              markersize = 10,
              linestyle='None')
        

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression,LassoLarsCV

from sklearn.preprocessing import PolynomialFeatures

p1 = PolynomialFeatures(degree=8)

p1.fit(data2[1:,0:1].reshape(-1,1))

x_h = p1.transform(data2[1:,0:1])

model = LassoLarsCV(cv=5)

# model = LinearRegression()


model.fit(x_h,data2[1:,1])

models.append(model)

trans.append(p1)

print(model.coef_)

print(r2_score(data2[1:,1],model.predict(x_h)))

xx = np.linspace(data2[1,0], data2[-1,0], 50).reshape(-1,1)

x_h = p1.transform(xx)

ax1.plot(xx,np.where(model.predict(x_h)<0,0,model.predict(x_h)),c=acid_color[1],linewidth=7)

ax1.plot([0,data2[1,0]],[0,0],c=acid_color[1],linewidth=5)


models_for_adsorption.append(model)

p1_for_adsorption.append(p1)




x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])

# bb = sigmoid(x, *models[0]).reshape([-1,1])

bb = models[0].predict(trans[0].transform(x.reshape([-1,1]))).reshape([-1,1])

bb = np.where(bb<0,0,bb)

x_= x.copy()

ff = np.vstack([0*x_[np.where(x_<data2[1,0])].reshape([-1,1]), 
                models[1].predict(trans[1].transform(x_[np.where(x_>=data2[1,0])].reshape([-1,1]))).reshape([-1,1])])

# ff = x_*0

ff = np.where(ff<0,0,ff)

tmp = bb + ff


d_x = [np.array([0])]

d_y = [np.array([0])]

for i in range(1,x.shape[0]):
    
    d_x.append(x[i])
    
    d_y.append((tmp[i]-tmp[i-1])/(x[i]-x[i-1]))



ax2.plot(d_x,d_y,
              # yerr=0,
              c=acid_color[2],
              label=acid_labels[2],
              marker=marker_style[2],
              markersize = 0,
              linewidth=7,
              linestyle='-')




tmp = tmp + 2 

for i in data1[:,0]:
    
    tmp[np.where(x>i)] = tmp[np.where(x>i)] - 1.5/1000




ax1.set_xlabel('Time (h)',fontproperties = 'Arial',size=25)

ax1.set_ylabel('Capacity (L)',fontproperties = 'Arial',size=25)

# plt.xticks(data2[:,0],data2[:,0].astype('int'),fontproperties = 'Arial',size=25) 

ax2.set_ylabel('Velocity(L/h)',fontproperties = 'Arial',size=25)

ax1.legend(prop={'family' : 'Arial', 'size' : 30},frameon=False, title='', loc = 7)

ax2.legend(prop={'family' : 'Arial', 'size' : 30},frameon=False, title='  \n\n\n\n\n\n\n\n')

ax1.tick_params(labelsize=20)

ax2.tick_params(labelsize=20) 






ax1 = ax[1]

ax2 = ax1.twinx()

ax1.set_title('b',fontproperties = 'Arial',size=50,loc='left', x = -0.15)

ax1.plot(x,tmp,
              # yerr=0,
              c=acid_color[3],
              label=acid_labels[3],
              marker=marker_style[3],
              markersize = 0,
              linewidth=7,
              linestyle='-')


ax2.plot(x,d_y/tmp,
              # yerr=0,
              c=acid_color[4],
              label=acid_labels[4],
              marker=marker_style[4],
              markersize = 0,
              linewidth=7,
              linestyle='-')

ax1.set_xlabel('Time (h)',fontproperties = 'Arial',size=25)

ax1.set_ylabel('Capacity (L)',fontproperties = 'Arial',size=25)

# plt.xticks(data2[:,0],data2[:,0].astype('int'),size=25) 

ax2.set_ylabel('Dilution Rate (1/h)',fontproperties = 'Arial',size=25)

ax1.legend(prop={'family' : 'Arial', 'size' : 30},frameon=False, title=' ')

ax2.legend(prop={'family' : 'Arial', 'size' : 30},frameon=False, title='  \n\n\n\n')

ax1.tick_params(labelsize=20)

ax2.tick_params(labelsize=20)


plt.show()






ext = []



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random


seed = 577

np.random.seed(seed)
random.seed(seed)

mpl.rc('axes', lw=2)

data_ = np.array(pd.read_excel('./exp/g3.xlsx',header=None))

data = data_[:-1,:].astype('float')

acid_labels = ['Glycerol','1,3-PDO','OD','Formic Acid','Acetic Acid','Lactic Acid','Butyric Acid','Base','Feed Medium']

acid_color = ['blue','red','black','red','blue','gray','green','orange','hotpink']

marker_style = ['^','o','s','o','^','s','v','*','>']

acid_top = [3.5,4,3,10]

acid_low = [2.5,3,2,9]


# plt.figure(dpi=300)

plt.rcParams['xtick.direction'] = 'in'

plt.rcParams['ytick.direction'] = 'in'

x = np.linspace(1,data.shape[0],data.shape[0])


fig,ax = plt.subplots(1,2,figsize=(25,10),dpi=300)

ax1 = ax[0]

ax2 = ax1.twinx()

ax1.set_title('a',fontproperties = 'Arial',size=50,loc='left', x = -0.15)

for i in range(3):
    
    
    if not(i >= 2):
            
        ax1.errorbar(x,data[:,i+1],
                     yerr=0,
                     c=acid_color[i],
                     label=acid_labels[i],
                     marker=marker_style[i],
                     markersize = 10,
                     linestyle='-')
        
        
    
    
    
    
    
    else:
        
        
        
        ax2.errorbar(x,data[:,i+1],
                     yerr=0,
                     c=acid_color[i],
                     label=acid_labels[i],
                     marker=marker_style[i],
                     markersize = 10,
                     linestyle='-')

ax1.set_xlabel('Time (h)',fontproperties = 'Arial',size=25)

ax1.set_ylabel('Concentration (g/L)',fontproperties = 'Arial',size=25)

# plt.xticks(x,data[:,0].astype('int'),size=25) 

ax2.set_ylabel('OD (A)',fontproperties = 'Arial',size=25)

ax1.legend(prop={'family' : 'Arial', 'size' : 30},frameon=False, title=' ')

ax2.legend(prop={'family' : 'Arial', 'size' : 30},frameon=False, title='  \n\n')

ax1.tick_params(labelsize=20)

ax2.tick_params(labelsize=20)

ax1 = ax[1]

ax2 = ax1.twinx()

ax1.set_title('b',fontproperties = 'Arial',size=50,loc='left', x = -0.15)

for i in range(3,9):
    
    
    if not(i >= 7):
            
        ax1.errorbar(x,data[:,i+1],
                     yerr=0,
                     c=acid_color[i],
                     label=acid_labels[i],
                     marker=marker_style[i],
                     markersize = 10,
                     linestyle='-')
    
    
    
    
    
    else:
        
        
        
        ax2.errorbar(x,data[:,i+1],
                     yerr=0,
                     c=acid_color[i],
                     label=acid_labels[i],
                     marker=marker_style[i],
                     markersize = 10,
                     linestyle='-')        
        


ax1.set_xlabel('Time (h)',fontproperties = 'Arial',size=25)

ax1.set_ylabel('Concentration (g/L)',fontproperties = 'Arial',size=25)

# plt.xticks(x,data[:,0].astype('int'),fontproperties = 'Arial',size=25) 

ax2.set_ylabel('Capacity (mL)',fontproperties = 'Arial',size=25)

ax1.legend(prop={'family' : 'Arial', 'size' : 30},frameon=False)

ax2.legend(prop={'family' : 'Arial', 'size' : 30},frameon=False)

ax1.tick_params(labelsize=20)

ax2.tick_params(labelsize=20)  

plt.show()


# from scipy.optimize import curve_fit

# def sigmoid(x, a, b):
    
#     y = 1 / (1 + np.exp(-b*(x-a)))

#     return y


# popt, pcov = curve_fit(sigmoid, data[:,0:1],data[:,8:9])

# x = np.linspace(0, 30, 50)

# y = sigmoid(x, *popt)

# # for i in 



from sklearn.metrics import r2_score



fig,ax = plt.subplots(1,2,figsize=(25,10),dpi=300)

ax1 = ax[0]

ax2 = ax1.twinx()

ax1.set_title('a',fontproperties = 'Arial',size=50,loc='left', x = -0.15)

for i in range(3):
    



    if i==0:
        
        
        # from sklearn.linear_model import LinearRegression,LassoLarsCV

        # from sklearn.preprocessing import PolynomialFeatures

        # p11 = PolynomialFeatures(degree=4)

        # p11.fit(data[0:10,0:1].reshape(-1,1))

        # x_h = p11.transform(data[0:10,0:1])
        
        # model1 = LinearRegression()

        # # model1 = LassoLarsCV(cv=5)

        # model1.fit(x_h,data[0:10,i+1])
        
        # print(model1.coef_)
        
        # print(r2_score(data[0:10,i+1],model1.predict(x_h)))
        
        # # print(model1.predict(p1.transform([[5]]))[0])

        # # print(model1.predict(p1.transform([[6]]))[0])

        # # print(model1.predict(p1.transform([[7]]))[0])


        # xx = np.linspace(0, 11, 50).reshape(-1,1)
        
        # x_h = p11.transform(xx)

        # ax1.plot(xx,model1.predict(x_h),c=acid_color[i],linewidth=5)
        




        # from sklearn.linear_model import LinearRegression,LassoLarsCV

        # from sklearn.preprocessing import PolynomialFeatures

        # p22 = PolynomialFeatures(degree=6)

        # p22.fit(data[10:,0:1].reshape(-1,1))

        # x_h = p22.transform(data[10:,0:1])
        
        # model2 = LinearRegression()

        # # model2 = LassoLarsCV(cv=5)

        # model2.fit(x_h,data[10:,i+1])
        
        # print(model2.coef_)
        
        # print(r2_score(data[10:,i+1],model2.predict(x_h)))
        
        # # print(model2.predict(p1.transform([[5]]))[0])

        # # print(model2.predict(p1.transform([[6]]))[0])

        # # print(model2.predict(p1.transform([[7]]))[0])


        # xx = np.linspace(11, data[-1,0], 50).reshape(-1,1)
        
        # x_h = p22.transform(xx)

        # ax1.plot(xx,model2.predict(x_h),c=acid_color[i],linewidth=5)


        
        
        # xxx = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
        
        # x = xxx[0:450,:]
        
        # x_h = p11.transform(x)
        
        # y = model1.predict(x_h)
        
        # y1 = np.where(y<0,0,y)
        
        
        # x = xxx[450::,:]
        
        # x_h = p22.transform(x)
        
        # y = model2.predict(x_h)
        
        # y2 = np.where(y<0,0,y)
        
        # y = np.hstack([y1,y2])

        
        # ext.append(y.reshape([-1,1]))


        
            
        # ax1.errorbar(data[:,0],data[:,i+1],
        #               yerr=0,
        #               c=acid_color[i],
        #               label=acid_labels[i],
        #               marker=marker_style[i],
        #               markersize = 10,
        #               linestyle='None')
        
        
        
        from sklearn.linear_model import LinearRegression,LassoLarsCV

        from sklearn.preprocessing import PolynomialFeatures
        
        
        from scipy.interpolate import interp1d,barycentric_interpolate




        p1 = PolynomialFeatures(degree=10)

        p1.fit(data[:,0:1].reshape(-1,1))

        x_h = p1.transform(data[:,0:1])
        
        model = LinearRegression()

        # model = LassoLarsCV(cv=5)

        model.fit(x_h,data[:,i+1])
        
        print(model.coef_)
        
        print(r2_score(data[:,i+1],model.predict(x_h)))
        

        xx = np.linspace(0, data[-1,0], 50).reshape(-1,1)
        
        x_h = p1.transform(xx)

        ax1.plot(xx,model.predict(x_h),c=acid_color[i],linewidth=5)
        

        x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
        
        x_h = p1.transform(x)
        
        y = model.predict(x_h)
        
        y = np.where(y<0,0,y)
        
        ext.append(y.reshape([-1,1]))


        models_for_adsorption.append(model)
        
        p1_for_adsorption.append(p1)



        
        # model = interp1d(data[:,0] ,data[:,i+1],kind=5,axis=0)
        
        # # print(model.coef_)
        
        # print(r2_score(data[:,i+1],model(data[:,0:1])))
        

        # xx = np.linspace(0, data[-1,0], 50)
        

        # ax1.plot(xx,model(xx),c=acid_color[i],linewidth=5)
        

        # x = np.linspace(0,data1[-1,0],1000)
        
        
        # y = model(x)
        
        # y = np.where(y<0,0,y)
        
        # ext.append(y.reshape([-1,1]))


        



        
        
        
        '''
        
        # cubic inte
        
        t = data[:,0:1]
        
        model = interp1d(t.reshape(t.shape[0],),data[:,i+1].reshape(t.shape[0],),kind=1,axis=0)
        
        # plt.plot(ode_t,model(ode_t))
        
        # plt.scatter(t,y[:,i])

        # plt.show()

        # ext.append(model(ode_t))

        
        xx = np.linspace(0, data[-1,0], 50).reshape(-1,1)
        
        ax1.plot(xx,model(xx),c=acid_color[i],linewidth=5)
        

        x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
        
        y = model(x)
        
        y = np.where(y<0,0,y)
        
        ext.append(y.reshape([-1,1]))

        '''        
        
        



        # if i==0:
            
        #     data[0,i+1] = 48.1637
            
        #     data[1,i+1] = 44.11151
            
        ax1.errorbar(data[:,0],data[:,i+1],
                      yerr=0,
                      c=acid_color[i],
                      label=acid_labels[i],
                      marker=marker_style[i],
                      markersize = 10,
                      linestyle='None')



    
    elif i == 1:
        
        
        from sklearn.linear_model import LinearRegression,LassoLarsCV

        from sklearn.preprocessing import PolynomialFeatures

        p1 = PolynomialFeatures(degree=10)

        p1.fit(data[:,0:1].reshape(-1,1))

        x_h = p1.transform(data[:,0:1])

        # model = LassoLarsCV(cv=5)
        
        model = LinearRegression()

        model.fit(x_h,data[:,i+1])
        
        print(model.coef_)
        
        print(r2_score(data[:,i+1],model.predict(x_h)))
        
        # print(model.predict(p1.transform([[5]]))[0])

        # print(model.predict(p1.transform([[6]]))[0])

        # print(model.predict(p1.transform([[7]]))[0])


        xx = np.linspace(0, data[-1,0], 50).reshape(-1,1)
        
        x_h = p1.transform(xx)

        ax1.plot(xx,model.predict(x_h),c=acid_color[i],linewidth=5)
        
        
        
        
        x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
        
        x_h = p1.transform(x)
        
        y = model.predict(x_h)
        
        y = np.where(y<0,0,y)
        
        ext.append(y.reshape([-1,1]))


        
            
        ax1.errorbar(data[:,0],data[:,i+1],
                      yerr=0,
                      c=acid_color[i],
                      label=acid_labels[i],
                      marker=marker_style[i],
                      markersize = 10,
                      linestyle='None')
        
        models_for_adsorption.append(model)
        
        p1_for_adsorption.append(p1)

    
        # from scipy.optimize import curve_fit

        # def sigmoid(x, a, b, c):
        
        #     y = c / (1 + np.exp(-b*(x-a)))
        
        #     return y
        
        
        # popt, pcov = curve_fit(sigmoid, data[:,0],data[:,i+1],bounds=([0,0,0], [20,10,50]))
        
        # print(popt)
        
        # print(r2_score(data[:,i+1],sigmoid(data[:,0], *popt)))

        # xx = np.linspace(0, data1[-1,0], 50)
        
        # yy = sigmoid(xx, *popt)
        
        # ax1.plot(xx,yy,c=acid_color[i],linewidth=5)
        
        # # from sklearn.linear_model import LinearRegression,LassoLarsCV

        # # from sklearn.preprocessing import PolynomialFeatures

        # # p1 = PolynomialFeatures(degree=5)

        # # p1.fit(data[:,0:1].reshape(-1,1))

        # # x_h = p1.transform(data[:,0:1])

        # # model = LassoLarsCV(cv=5)

        # # model.fit(x_h,data[:,i+1])
        
        # # print(model.coef_)
        
        # # print(r2_score(data[:,i+1],model.predict(x_h)))
        
        
        # # print(model.predict(p1.transform([[5]]))[0])

        # # print(model.predict(p1.transform([[6]]))[0])

        # # print(model.predict(p1.transform([[7]]))[0])

        
        # # xx = np.linspace(0, 30, 50).reshape(-1,1)
        
        # # x_h = p1.transform(xx)

        # # ax2.plot(xx,model.predict(x_h),c=acid_color[i],linewidth=3)
        
        
        
        # x = np.linspace(0,data1[-1,0],1000)
        
        # # x_h = p1.transform(x)
        
        # # y = model.predict(x_h)
        
        # y = sigmoid(x, *popt)
        
        # y = np.where(y<0,0,y)

        
        # ext.append(y.reshape([-1,1]))
        
        
        
        # ax1.errorbar(data[:,0],data[:,i+1],
        #              yerr=0,
        #              c=acid_color[i],
        #              label=acid_labels[i],
        #              marker=marker_style[i],
        #              markersize = 10,
        #              linestyle='None')
    
    
    
    
    else:
        
        
        
        from scipy.optimize import curve_fit

        # def sigmoid(x, a, b, c):
        
        #     y = c / (1 + np.exp(-b*(x-a)))
        
        #     return y
        
        
        # popt, pcov = curve_fit(sigmoid, data[:,0],data[:,i+1],bounds=([0,0,0], [20,10,50]))
        
        # print(popt)
        
        # print(r2_score(data[:,i+1],sigmoid(data[:,0], *popt)))

        # xx = np.linspace(0, data1[-1,0], 50)
        
        # yy = sigmoid(xx, *popt)
        
        # ax2.plot(xx,yy,c=acid_color[i],linewidth=5)
        
        from sklearn.linear_model import LinearRegression,LassoLarsCV

        from sklearn.preprocessing import PolynomialFeatures

        p1 = PolynomialFeatures(degree=10)

        p1.fit(data[:,0:1].reshape(-1,1))

        x_h = p1.transform(data[:,0:1])

        # model = LassoLarsCV(cv=5)
        
        model = LinearRegression()

        model.fit(x_h,data[:,i+1])
        
        print(model.coef_)
        
        print(r2_score(data[:,i+1],model.predict(x_h)))
        
        
        # print(model.predict(p1.transform([[5]]))[0])

        # print(model.predict(p1.transform([[6]]))[0])

        # print(model.predict(p1.transform([[7]]))[0])

        
        xx = np.linspace(0, data1[-1,0], 50).reshape(-1,1)
        
        x_h = p1.transform(xx)

        ax2.plot(xx,model.predict(x_h),c=acid_color[i],linewidth=5)
        
        
        
        x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
        
        x_h = p1.transform(x)
        
        y = model.predict(x_h)
        
        
        # x = np.linspace(0,data1[-1,0],1000)
        
        # y = sigmoid(x, *popt)
        
        # y = np.where(y<0,0,y)

        
        ext.append(y.reshape([-1,1]))
        
        
        
        ax2.errorbar(data[:,0],data[:,i+1],
                      yerr=0,
                      c=acid_color[i],
                      label=acid_labels[i],
                      marker=marker_style[i],
                      markersize = 10,
                      linestyle='None')

        # models_for_adsorption.append(model)
        
        # p1_for_adsorption.append(p1)


ax1.set_xlabel('Time (h)',fontproperties = 'Arial',size=25)

ax1.set_ylabel('Concentration (g/L)',fontproperties = 'Arial',size=25)

# plt.xticks(data[:,0],data[:,0].astype('int'),size=25) 



ax2.set_ylabel('OD (A)',fontproperties = 'Arial',size=25)

ax1.legend(prop={'family' : 'Arial', 'size' : 30},frameon=False, title=' \n\n\n\n')

ax2.legend(prop={'family' : 'Arial', 'size' : 30},frameon=False, title=' ')

ax1.tick_params(labelsize=20)

ax2.tick_params(labelsize=20)

ax1 = ax[1]

# ax2 = ax1.twinx()

ax1.set_title('b',fontproperties = 'Arial',size=50,loc='left', x = -0.15)



for i in range(3,9):
    
    if i == 3:
        
        
        from sklearn.linear_model import LinearRegression,LassoLarsCV
         
        from sklearn.preprocessing import PolynomialFeatures
         
        p1 = PolynomialFeatures(degree=6)
         
        p1.fit(data[:,0:1].reshape(-1,1))
         
        x_h = p1.transform(data[:,0:1])
         
        model = LassoLarsCV(cv=5)
            
        # model = LinearRegression()
         
        model.fit(x_h,data[:,i+1])
            
        print(model.coef_)
            
        print(r2_score(data[:,i+1],model.predict(x_h)))
            
        # print(model.predict(p1.transform([[5]]))[0])
         
        # print(model.predict(p1.transform([[6]]))[0])
         
        # print(model.predict(p1.transform([[7]]))[0])
         
         
        xx = np.linspace(0, data[-1,0], 50).reshape(-1,1)
            
        x_h = p1.transform(xx)
         
        ax1.plot(xx,model.predict(x_h),c=acid_color[i],linewidth=5)
            
            
            
            
        x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
            
        x_h = p1.transform(x)
            
        y = model.predict(x_h)
            
        y = np.where(y<0,0,y)
            
        ext.append(y.reshape([-1,1]))

          
          
          
          
          
        ax1.errorbar(data[:,0],data[:,i+1],
               yerr=0,
               c=acid_color[i],
               label=acid_labels[i],
               marker=marker_style[i],
               markersize = 10,
               linestyle='None')

        models_for_adsorption.append(model)
        
        p1_for_adsorption.append(p1)

    
    
    elif i == 4:
        
        
        from sklearn.linear_model import LinearRegression,LassoLarsCV
         
        from sklearn.preprocessing import PolynomialFeatures
         
        p1 = PolynomialFeatures(degree=8)
         
        p1.fit(data[:,0:1].reshape(-1,1))
         
        x_h = p1.transform(data[:,0:1])
         
        # model = LassoLarsCV(cv=5)
            
        model = LinearRegression()
         
        model.fit(x_h,data[:,i+1])
            
        print(model.coef_)
            
        print(r2_score(data[:,i+1],model.predict(x_h)))
            
        # print(model.predict(p1.transform([[5]]))[0])
         
        # print(model.predict(p1.transform([[6]]))[0])
         
        # print(model.predict(p1.transform([[7]]))[0])
         
         
        xx = np.linspace(0, data[-1,0], 50).reshape(-1,1)
            
        x_h = p1.transform(xx)
         
        ax1.plot(xx,model.predict(x_h),c=acid_color[i],linewidth=5)
            
            
            
            
        x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
            
        x_h = p1.transform(x)
            
        y = model.predict(x_h)
            
        y = np.where(y<0,0,y)
            
        ext.append(y.reshape([-1,1]))

          
          
          
          
          
        ax1.errorbar(data[:,0],data[:,i+1],
               yerr=0,
               c=acid_color[i],
               label=acid_labels[i],
               marker=marker_style[i],
               markersize = 10,
               linestyle='None')

    
        models_for_adsorption.append(model)
        
        p1_for_adsorption.append(p1)
    
    elif i == 5:
        
        
        from sklearn.linear_model import LinearRegression,LassoLarsCV
         
        from sklearn.preprocessing import PolynomialFeatures
         
        p1 = PolynomialFeatures(degree=10)
         
        p1.fit(data[:,0:1].reshape(-1,1))
         
        x_h = p1.transform(data[:,0:1])
         
        # model = LassoLarsCV(cv=5)
            
        model = LinearRegression()
         
        model.fit(x_h,data[:,i+1])
            
        print(model.coef_)
            
        print(r2_score(data[:,i+1],model.predict(x_h)))
            
        # print(model.predict(p1.transform([[5]]))[0])
         
        # print(model.predict(p1.transform([[6]]))[0])
         
        # print(model.predict(p1.transform([[7]]))[0])
         
         
        xx = np.linspace(0, data[-1,0], 50).reshape(-1,1)
            
        x_h = p1.transform(xx)
         
        ax1.plot(xx,model.predict(x_h),c=acid_color[i],linewidth=5)
            
            
            
            
        x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
            
        x_h = p1.transform(x)
            
        y = model.predict(x_h)
            
        y = np.where(y<0,0,y)
            
        ext.append(y.reshape([-1,1]))

          
          
          
          
          
        ax1.errorbar(data[:,0],data[:,i+1],
               yerr=0,
               c=acid_color[i],
               label=acid_labels[i],
               marker=marker_style[i],
               markersize = 10,
               linestyle='None')
        
        models_for_adsorption.append(model)
        
        p1_for_adsorption.append(p1)

    
    
    elif i == 6:
        
        
        from sklearn.linear_model import LinearRegression,LassoLarsCV
         
        from sklearn.preprocessing import PolynomialFeatures
         
        p1 = PolynomialFeatures(degree=10)
         
        p1.fit(data[:,0:1].reshape(-1,1))
         
        x_h = p1.transform(data[:,0:1])
         
        # model = LassoLarsCV(cv=5)
            
        model = LinearRegression()
         
        model.fit(x_h,data[:,i+1])
            
        print(model.coef_)
            
        print(r2_score(data[:,i+1],model.predict(x_h)))
            
        # print(model.predict(p1.transform([[5]]))[0])
         
        # print(model.predict(p1.transform([[6]]))[0])
         
        # print(model.predict(p1.transform([[7]]))[0])
         
         
        xx = np.linspace(0, data[-1,0], 50).reshape(-1,1)
            
        x_h = p1.transform(xx)
         
        ax1.plot(xx,model.predict(x_h),c=acid_color[i],linewidth=5)
            
            
            
            
        x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
            
        x_h = p1.transform(x)
            
        y = model.predict(x_h)
            
        y = np.where(y<0,0,y)
            
        ext.append(y.reshape([-1,1]))

          
          
          
          
          
        ax1.errorbar(data[:,0],data[:,i+1],
               yerr=0,
               c=acid_color[i],
               label=acid_labels[i],
               marker=marker_style[i],
               markersize = 10,
               linestyle='None')
        
        models_for_adsorption.append(model)
        
        p1_for_adsorption.append(p1)

    

    # elif not(i >= 7):
        
    #     from scipy.optimize import curve_fit

    #     def sigmoid(x, a, b, c):
      
    #         y = c / (1 + np.exp(-b*(x-a)))
      
    #         return y
      
      
    #     popt, pcov = curve_fit(sigmoid, data[:,0],data[:,i+1],bounds=([0,0,0], [50,10,50]))
      
    #     print(popt)
      
    #     print(r2_score(data[:,i+1],sigmoid(data[:,0], *popt)))
      
    #     xx = np.linspace(0, data[-1,0], 50)
      
    #     yy = sigmoid(xx, *popt)
      
      
      
      
    #     ax1.plot(xx,yy,c=acid_color[i],linewidth=5)
      
      
    #     x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
      
    #     y = sigmoid(x, *popt)
      
    #     y = np.where(y<0,0,y)
      
    #     ext.append(y.reshape([-1,1]))
           
           
           
          
          
          
    #     # from sklearn.linear_model import LinearRegression,LassoLarsCV
         
    #     # from sklearn.preprocessing import PolynomialFeatures
         
    #     # p1 = PolynomialFeatures(degree=6)
         
    #     # p1.fit(data[:,0:1].reshape(-1,1))
         
    #     # x_h = p1.transform(data[:,0:1])
         
    #     # model = LassoLarsCV(cv=5)
            
    #     # # model = LinearRegression()
         
    #     # model.fit(x_h,data[:,i+1])
            
    #     # print(model.coef_)
            
    #     # print(r2_score(data[:,i+1],model.predict(x_h)))
            
    #     # # print(model.predict(p1.transform([[5]]))[0])
         
    #     # # print(model.predict(p1.transform([[6]]))[0])
         
    #     # # print(model.predict(p1.transform([[7]]))[0])
         
         
    #     # xx = np.linspace(0, data[-1,0], 50).reshape(-1,1)
            
    #     # x_h = p1.transform(xx)
         
    #     # ax1.plot(xx,model.predict(x_h),c=acid_color[i],linewidth=5)
            
            
            
            
    #     # x = np.linspace(0,data1[-1,0],1000).reshape([-1,1])
            
    #     # x_h = p1.transform(x)
            
    #     # y = model.predict(x_h)
            
    #     # y = np.where(y<0,0,y)
            
    #     # ext.append(y.reshape([-1,1]))

          
          
          
          
          
    #     ax1.errorbar(data[:,0],data[:,i+1],
    #             yerr=0,
    #             c=acid_color[i],
    #             label=acid_labels[i],
    #             marker=marker_style[i],
    #             markersize = 10,
    #             linestyle='None')
      
        
    
    
ax1.set_xlabel('Time (h)',fontproperties = 'Arial',size=25)

ax1.set_ylabel('Concentration (g/L)',fontproperties = 'Arial',size=25)

# plt.xticks(data[:,0],data[:,0].astype('int'),fontproperties = 'Arial',size=25) 

# ax2.set_ylabel('Capacity (mL)',fontproperties = 'Arial',size=25)

ax1.legend(prop={'family' : 'Arial', 'size' : 25},frameon=False)

# ax2.legend(prop={'family' : 'Arial', 'size' : 20},frameon=False)

ax1.tick_params(labelsize=20)

# ax2.tick_params(labelsize=20)  

plt.show()



import scipy.optimize as opt


parameters = [



[0.16722678, 0.16442365, 0.05086458, 0.16542762, 0.05074939, 0.13561252, 0.0976876 , 0.1707491 ],

[0.03352138, 0.07404045, 0.05101445, 0.06060472, 0.04333584, 0.01758684, 0.03256742, 0.07389156],

[0.16637952, 0.92536901, 0.0805736 , 0.49922464, 0.21616394, 0.0898183 , 0.61580155, 0.58438589],

[0.41403644, 0.51241312, 0.07583296, 0.89605222, 0.2117562 , 0.7098126 , 0.18924946, 0.56843274],

[0.10243063, 0.12034347, 0.08146878, 0.1490835 , 0.10936842, 0.04015449, 0.16495753, 0.03003341],


[0.31869627, 6.81279379, 0.40771035, 0.32298684, 0.0760788 , 0.95405152, 0.0408935 , 0.09731724],


]


simulate_t = np.linspace(0,data1[-1,0],1000)

delta_t = (simulate_t[1::] - simulate_t[0:-1]).mean()

r = 1.0

delta_r = 0.01

simulate_r = np.arange(0, r + delta_t, delta_r)

data_array = np.zeros([simulate_t.shape[0],simulate_r.shape[0]+2])



def goal(xx):
    
    print(xx)
    

    # parames definition
    
    # Li_r
    
    q_m = xx[0]
    
    K_a = xx[1]
    
    
    # C_r
    
    m = 30
    
    v = 3
    
    s = xx[2]
    
    k_f =  xx[3] #KL
    
    # Car_r
    
    e_p = xx[4] # fai*e
    
    p_a = xx[5]
    
    D_p = xx[6]
    
    D_s = xx[7]

    
    
    
    # init_data
    
    data_array[0,-2] = models_for_adsorption[index_adsoprtion].predict((p1_for_adsorption[index_adsoprtion].transform([[0]])))[0]
    
    
    
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
            
            
            # implicit
            
            # cpr_t_1 = x[1:-1]
            
            # cpr_t = data_array_t[1:-2]
            
            # cpr_t_s_1 = data_array_t[2:-1]
            
            # cpr_t_s__1 = data_array_t[0:-3]
            
            # r_ = simulate_r[1:-1]
            
            # diff_t = (e_p+(p_a*q_m*K_a)/((1+K_a*cpr_t)**2))*((cpr_t_1-cpr_t)/delta_t)
            
            # eq_right = (D_p+(D_s*p_a*q_m*K_a)/((1+K_a*cpr_t)**2))*((cpr_t_s_1-2*cpr_t+cpr_t_s__1)/(delta_r**2)) - ((2*D_s*p_a*q_m*(K_a**2))/((1+K_a*cpr_t)**3))*(((cpr_t_s_1-cpr_t)/(delta_r))**2) + (2/r_)*(D_p+(D_s*p_a*q_m*K_a)/((1+K_a*cpr_t)**2))*((cpr_t_s_1-cpr_t)/delta_r)
            
            
            # explicit
            
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
        
        if index_adsoprtion != 2:
            
        
            v = 3 + models_for_adsorption[0].predict((p1_for_adsorption[0].transform([[simulate_t[i]]])))[0] + models_for_adsorption[1].predict((p1_for_adsorption[1].transform([[simulate_t[i]]])))[0]
            
            data_array_t = data_array[i-1,0:-1]
            
            CA = max( data_array[i-1,-2] - max(m*s*k_f*(data_array[i-1,-2]-data_array[i-1,-3])*delta_t/v, 0), 0)
            
            data_array[i,-1] = min(max(models_for_adsorption[index_adsoprtion].predict((p1_for_adsorption[index_adsoprtion].transform([[simulate_t[i]]])))[0],0),CA)
            
            data_array[i,-2] = max(models_for_adsorption[index_adsoprtion].predict((p1_for_adsorption[index_adsoprtion].transform([[simulate_t[i]]])))[0],0)
            
            data_array[i,0:-2] = pvsd(data_array_t, CA , q_m, K_a, m, v, s, k_f, e_p, p_a, D_p, D_s)
            
        
        elif index_adsoprtion == 2:
            
            v = 3 + models_for_adsorption[0].predict((p1_for_adsorption[0].transform([[simulate_t[i]]])))[0] + models_for_adsorption[1].predict((p1_for_adsorption[1].transform([[simulate_t[i]]])))[0]
            
            data_array_t = data_array[i-1,0:-1]
            
            CA = max( data_array[i-1,-2] - max(m*s*k_f*(data_array[i-1,-2]-data_array[i-1,-3])*delta_t/v, 0), 0)
            
            data_array[i,-1] = CA
            
            data_array[i,-2] = max(models_for_adsorption[index_adsoprtion].predict((p1_for_adsorption[index_adsoprtion].transform([[simulate_t[i]]])))[0],0)
            
            data_array[i,0:-2] = pvsd(data_array_t, CA , q_m, K_a, m, v, s, k_f, e_p, p_a, D_p, D_s)

            
        
        
    
    plt.plot(simulate_t, data_array[:,-1])
        
    plt.show()
    


adsorption_term = []

for i in range(len(parameters)):

    index_adsoprtion = i + 2
    
    goal(parameters[i])

    d = (data_array[1::,-1]-data_array[0:-1,-2])
    
    d = np.where(d>0,0,d)
    
    print(d.sum())
    
    adsorption_term.append(d)

adsorption_term = np.array(adsorption_term)

adsorption_term = np.hstack([np.zeros([6,1]),adsorption_term])

# plt.xlabel('Time (h)')
# plt.ylabel('C (g/L)')
# plt.title('Glycerol')
# plt.savefig('adsorption-Glycerol.png',dpi=300)
# plt.close()

# plt.plot(simulate_t,-adsorption_term[1])
# plt.xlabel('Time (h)')
# plt.ylabel('C (g/L)')
# plt.title('PDO')
# plt.savefig('adsorption-PDO.png',dpi=300)
# plt.close()

# plt.plot(simulate_t,-adsorption_term[2])
# plt.xlabel('Time (h)')
# plt.ylabel('C (g/L)')
# plt.title('FA')
# plt.savefig('adsorption-FA.png',dpi=300)
# plt.close()

# plt.plot(simulate_t,-adsorption_term[3])
# plt.xlabel('Time (h)')
# plt.ylabel('C (g/L)')
# plt.title('AA')
# plt.savefig('adsorption-AA.png',dpi=300)
# plt.close()

# plt.plot(simulate_t,-adsorption_term[4])
# plt.xlabel('Time (h)')
# plt.ylabel('C (g/L)')
# plt.title('LA')
# plt.savefig('adsorption-LA.png',dpi=300)
# plt.close()

# plt.plot(simulate_t,-adsorption_term[5])
# plt.xlabel('Time (h)')
# plt.ylabel('C (g/L)')
# plt.title('BA')
# plt.savefig('adsorption-BA.png',dpi=300)
# plt.close()





D = d_y/tmp


ext[2] = ext[2]*0.4579 + 0.02

ext[2], ext[3], ext[4], ext[5], ext[6] = ext[3], ext[4], ext[5], ext[6], ext[2]

ext = np.array(ext).reshape([7,1000])

mass = [92.09,76.09,46.03,60.05,90.08,88.11]

for i in range(len(mass)):
    
    ext[i] = ext[i]/mass[i]*1000000000
    
    adsorption_term[i] = adsorption_term[i]/mass[i]*1000000000
    
    #mol/L
    
    # ext[i] = ext[i]/ext[-1]
    
d_ext = (ext[:,1::] - ext[:,0:-1])/(x[1::] - x[0:-1]).reshape([1,-1])

d_ext = np.hstack([np.zeros(7).reshape(7,1),d_ext])

# d_ext[0] = -d_ext[0]

A = np.array(pd.read_excel('./mod/cb304equation.xlsx',header=None))[0:11,0:15]

A[np.isnan(A)] = 0

A = -A


Ac = np.hstack([A[0:11,0:7],A[0:11,13:14]])

Am = np.hstack([A[0:11,7:13],A[0:11,14:15]])



vs = []

vr = []

for i in range(1000):
    
    r = d_ext[:,i]
    
    r = r + D[i,0]*ext[:,i]
    
    r[0] = r[0] - 1*5.42947116950809*D[i,0]*1000000000 + adsorption_term[0,i]
    
    
    for j in range(1,6):
        
        r[j] -= adsorption_term[j,i]
    
    
    
    
    r[0] = -r[0]
    
    
    r = r/ext[-1,i]
    
    r = r.reshape([7,1])
    
    # r = r[1::]
    
    #stupid!
    
    # goal = np.vstack([np.zeros(13).reshape([13,1]),
    #                   r[0:6,:].reshape([6,1]),
    #                   np.zeros(1).reshape([1,1]),
    #                   r[6:7,:].reshape([1,1])])
    
    # v = np.matmul(np.linalg.inv(np.matmul((A.T),A)),goal)
    
    #brilliant!
    
    v = -np.matmul(np.matmul(np.linalg.pinv(Ac),Am),r)
    
    vs.append(v)
    
    vr.append(r)
    

vs = np.array(vs)/1000000

vr = np.array(vr)

vr[:,0:-1] = vr[:,0:-1]/1000000


start_index = 0

for i in range(8):
    
    fig, ax = plt.subplots()
    
    ax.spines['top'].set_visible(False)
    
    ax.spines['right'].set_visible(False)
    
    plt.plot(x[start_index::],vs[start_index::,i],c='r')
    
    plt.savefig('g3-p' + str(i) + '.png',transparent=True,dpi=300)
    
    plt.show()



for i in range(7):
    
    fig, ax = plt.subplots()
    
    ax.spines['top'].set_visible(False)
    
    ax.spines['right'].set_visible(False)
    
    plt.plot(x[start_index::],vr[start_index::,i],c='r')
    
    plt.savefig('g3-p' + str(i+8) + '.png',transparent=True,dpi=300)
    
    plt.show()

np.savetxt('g3_vs.txt', vs.reshape([1000,8]))

np.savetxt('g3_vr.txt', vr.reshape([1000,7]))

np.savetxt('g3_x.txt',x)