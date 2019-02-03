# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:06:22 2019

@author: Yang Xu
"""

##-----------------------------------------------------------------------------
##import modules
from sklearn import mixture
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import pandas as pd
import seaborn as sns
import math

##-----------------------------------------------------------------------------
##import dataset and preperation
##Cautious: the task only deals with 2 classes classification problem
tr = pd.read_table('synth.tr',sep='\s+')
te = pd.read_table('synth.te',sep='\s+')

##-----------------------------------------------------------------------------
##estimate parameters using different cases
def para_est(train=tr,metrics="case1"):
    
    x = train.iloc[:, :-1].values
    y = train.iloc[:,-1].values
    ##case1 estimates parameters by assuming
    ##features are stasticially independent and they have the same variance
    mu0 = x[y==0].mean(axis=0)
    mu1 = x[y==1].mean(axis=0)
    if metrics=="case1":
        cov0 = np.cov(x[y==0].T)
        cov1 = np.cov(x[y==1].T)
        var = np.vstack((np.diagonal(cov0),np.diagonal(cov1))).mean()
        cov0 = np.array([[var,0],[0,var]]).copy()
        cov1 = np.array([[var,0],[0,var]]).copy()
        
    ##case2 estimates parameters by assuming
    ##the covariance matrices for all the classes are identical
    if metrics=="case2":
        cov0 = np.cov(x.T)
        cov1 = cov0.copy()
        
    ##case3 estimates parameters by assuming
    ##The covariance matrices are different from each category, and 
    ##decision boundary would be hyperquadratic for 2-D Gaussian
    if metrics=="case3":
        cov0 = np.cov(x[y==0].T)
        cov1 = np.cov(x[y==1].T)
    
    return [mu0, mu1, cov0,cov1]

##-----------------------------------------------------------------------------
##classify test dataset using different cases
def bayes_des_rule(para,test,prior=[0.5,0.5]):
    test=np.array(test)
    a = test-para[0]
    b = test-para[1]
    g0 = (-1/2)*((a.T*np.linalg.inv(para[2])*a).sum()+\
          math.log(np.linalg.det(para[2])))+math.log(prior[0])
    g1 = (-1/2)*((b.T*np.linalg.inv(para[3])*b).sum()+\
          math.log(np.linalg.det(para[3])))+math.log(prior[1])
    
    return [g0,g1].index(max(g0,g1)),g0-g1

##-----------------------------------------------------------------------------
##evualte predication
parameters = para_est(train=tr,metrics="case3")
prior0=np.linspace(0, 1)
prior0=prior0.tolist()
accuracy0=[]
accuracy1=[]
y_test=te['yc'].tolist()
for i in prior0:
    if i==0 or i==1 :
        continue
    prior_p = [i,1-i]
    y_pred=[]
    for index, row in te.iterrows():
        test = te.iloc[index, :-1].values
        prediction,g=bayes_des_rule(para=parameters,test=test,prior=prior_p)
        y_pred.append(prediction)
    
    accu=pd.crosstab(np.array(y_test), np.array(y_pred), \
                     rownames = ['y_test'],colnames = ['y_pred']).values
    accuracy0.append(accu[0,0]/(accu[0,1]+accu[0,0]))
    accuracy1.append(accu[1,1]/(accu[1,0]+accu[1,1]))    

plt.plot(prior0[1:len(prior0)-1], accuracy0,linewidth=2.0,label='Class 0')
plt.plot(prior0[1:len(prior0)-1], accuracy1,linewidth=2.0,label='Class 1')
plt.title(" ")
plt.xlabel("Class 0 prior probability")
plt.ylabel("Accuracy")
plt.legend(loc='lower left')
plt.show()

##-----------------------------------------------------------------------------
##plot decision boundary
X, Y = np.mgrid[-1:1:100j,-1:1:100j]
x = X.ravel()
y = Y.ravel()

para_case1 = para_est(train=tr,metrics="case1")
para_case2 = para_est(train=tr,metrics="case2")
para_case3 = para_est(train=tr,metrics="case3")

gs_c1 = []
for i in range(100**2):
    prediction,g=bayes_des_rule(para=para_case1,test=np.array((x[i],y[i])),\
                                prior=[0.5,0.5])
    gs_c1.append(g)
    
p_c1 = np.array(gs_c1).reshape(X.shape)

gs_c2 = []
for i in range(100**2):
    prediction,g=bayes_des_rule(para=para_case2,test=np.array((x[i],y[i])),\
                                prior=[0.5,0.5])
    gs_c2.append(g)
    
p_c2 = np.array(gs_c2).reshape(X.shape)

gs_c3 = []
for i in range(100**2):
    prediction,g=bayes_des_rule(para=para_case3,test=np.array((x[i],y[i])),\
                                prior=[0.5,0.5])
    gs_c3.append(g)
    
p_c3 = np.array(gs_c3).reshape(X.shape)

#te.insert(3,"pred",y_pred)
sns.lmplot(x="xs", y="ys", data=te, fit_reg=False, hue='yc', legend=False, \
           palette="Set1")
#sns.lmplot(x="xs", y="ys", data=te, fit_reg=False, hue='pred', legend=False, \
#           palette="Set1")
pl.contour(X, Y, p_c1, 0,label="case1")
pl.contour(X, Y, p_c2, 0,label="case2")
pl.contour(X, Y, p_c3, 0,label="case3")
plt.legend(loc='lower right')

##-----------------------------------------------------------------------------
##Bimodal Gaussian
##Next, use density estimation to estimate parameters of bimodal gaussian
##Use case 3 to construct decision rule
##hard coding for bimodal gaussian estiamation
xs = tr['xs']
ys = tr['ys']
sns.kdeplot(xs[tr.yc==0], label='xs_0', shade=True)
sns.kdeplot(xs[tr.yc==1], label='xs_1', shade=True)
sns.kdeplot(ys[tr.yc==0], label='ys_0', shade=True)
sns.kdeplot(ys[tr.yc==1], label='ys_1', shade=True)
x=tr.iloc[:,0].values

##density estimation for bimodal gaussian
dens=np.histogram(x[tr.yc==0], bins=20, \
                  range=None, normed=None, weights=None, density=None)
y=dens[0].tolist()
y.append(0)
xx=dens[1].tolist()
dens=np.array((xx,y)).T
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(dens)
mu_0_0=clf.means_[0][0]
mu_0_1=clf.means_[1][0]
sig_0_0=clf.covariances_[0][0][0]
sig_0_1=clf.covariances_[1][0][0]
##To simplify the issue, I use covariance of complete xs and ys to 
##build covariance matrix for each case
cov0=np.cov(xs[tr.yc==0],ys[tr.yc==0])
cov0_0=np.array([[sig_0_0,cov0[0,1]],[cov0[0,1],cov0[1,1]]])
cov0_1=np.array([[sig_0_1,cov0[0,1]],[cov0[0,1],cov0[1,1]]])

dens=np.histogram(x[tr.yc==1], bins=20, \
                  range=None, normed=None, weights=None, density=None)
y=dens[0].tolist()
y.append(0)
xx=dens[1].tolist()
dens=np.array((xx,y)).T
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(dens)
mu_1_0=clf.means_[0][0]
mu_1_1=clf.means_[1][0]
sig_1_0=clf.covariances_[0][0][0]
sig_1_1=clf.covariances_[1][0][0]
cov1=np.cov(xs[tr.yc==1],ys[tr.yc==1])
cov1_0=np.array([[sig_1_0,cov1[0,1]],[cov1[0,1],cov1[1,1]]])
cov1_1=np.array([[sig_1_1,cov1[0,1]],[cov1[0,1],cov1[1,1]]])

muy0 = ys[tr.yc==0].mean()
muy1 = ys[tr.yc==1].mean()

mu00 = np.array((mu_0_0,muy0))
mu01 = np.array((mu_0_1,muy0))
mu10 = np.array((mu_1_0,muy1))
mu11 = np.array((mu_1_1,muy1))

y_pred=[]
prior=[0.5,0.5]
for index, row in te.iterrows():
    test = te.iloc[index, :-1].values
    
    g0_0=(-1/2)*(((test-mu00).T*np.linalg.inv(cov0_0)*(test-mu00)).sum()+\
          math.log(np.linalg.det(cov0_0)))+math.log(prior[0])
    g0_1=(-1/2)*(((test-mu01).T*np.linalg.inv(cov0_1)*(test-mu01)).sum()+\
          math.log(np.linalg.det(cov0_1)))+math.log(prior[0])
    g1_0=(-1/2)*(((test-mu10).T*np.linalg.inv(cov1_0)*(test-mu10)).sum()+\
          math.log(np.linalg.det(cov1_0)))+math.log(prior[0])
    g1_1=(-1/2)*(((test-mu11).T*np.linalg.inv(cov1_1)*(test-mu11)).sum()+\
          math.log(np.linalg.det(cov1_1)))+math.log(prior[0])
    
    indx=[g0_0,g0_1,g1_0,g1_1].index(max(g0_0,g0_1,g1_0,g1_1))
    
    y_pred.append([0,0,1,1][indx])
    
y_pred_c3 = []
for index, row in te.iterrows():
    test = te.iloc[index, :-1].values
    prediction,g=bayes_des_rule(para=para_case3,test=test,\
                                prior=[0.5,0.5])
    y_pred_c3.append(prediction)
    
te.insert(3,"pred",y_pred)
te.insert(4,"pred_c3",y_pred_c3)
sns.lmplot(x="xs", y="ys", data=te, fit_reg=False, hue='yc', legend=False, \
           palette="Set1")
plt.legend(loc='lower right')
sns.lmplot(x="xs", y="ys", data=te, fit_reg=False, hue='pred', legend=False, \
           palette="Set1")
plt.legend(loc='lower right')
sns.lmplot(x="xs", y="ys", data=te, fit_reg=False, hue='pred_c3', legend=False, \
           palette="Set1")
plt.legend(loc='lower right')

pd.crosstab(np.array(y_test), np.array(y_pred), \
                     rownames = ['y_test'],colnames = ['y_pred'])
pd.crosstab(np.array(y_test), np.array(y_pred_c3), \
                     rownames = ['y_test'],colnames = ['y_pred_case3'])
