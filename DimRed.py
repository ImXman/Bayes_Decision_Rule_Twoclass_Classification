# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:38:23 2019

@author: Yang Xu
"""

##-----------------------------------------------------------------------------
##import modules
import numpy as np
import math
import timeit
import operator

##-----------------------------------------------------------------------------
##main functions

##estimate parameters using different cases
def para_est(x,y,metrics="case1"):
    
    ##case1 estimates parameters by assuming
    ##features are stasticially independent and they have the same variance
    mu0 = x[y==0].mean(axis=0)
    mu1 = x[y==1].mean(axis=0)
    if metrics=="case1":
        cov0 = np.cov(x[y==0].T)
        cov1 = np.cov(x[y==1].T)
        if x.shape[1]==1:
            var = (cov1+cov0)/2
        else:
            var = np.vstack((np.diagonal(cov0),np.diagonal(cov1))).mean()
        
        cov0 = np.zeros((x.shape[1],x.shape[1]))
        np.fill_diagonal(cov0,var)
        cov0=cov0.reshape(x.shape[1],x.shape[1])
        cov1 = cov0.copy()
        
    ##case2 estimates parameters by assuming
    ##the covariance matrices for all the classes are identical
    if metrics=="case2":
        cov0 = np.cov(x.T)
        cov0 = cov0.reshape(x.shape[1],x.shape[1])
        cov1 = cov0.copy()
        
    ##case3 estimates parameters by assuming
    ##The covariance matrices are different from each category, and 
    ##decision boundary would be hyperquadratic for 2-D Gaussian
    if metrics=="case3":
        cov0 = np.cov(x[y==0].T)
        cov0 = cov0.reshape(x.shape[1],x.shape[1])
        cov1 = np.cov(x[y==1].T)
        cov1 = cov1.reshape(x.shape[1],x.shape[1])
    
    return [mu0, mu1, cov0,cov1]

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

##data normalization
def normalization(data,mu=None,std=None):
    
    if mu is None and std is None:
        std=np.std(data,axis=0)
        std=std.reshape(1,len(std))
        mu=np.mean(data,axis=0)
        mu=mu.reshape(1,len(mu))
    norm_data = data.copy()
    norm_data= (norm_data-mu)/std
    
    return norm_data,[mu,std]

##data for PCA function doesn't contain the class label column
def PCA(data,error=0.1):
    
    ##calculate n dimension means
    pmu = np.mean(data,axis=0)
    pmu = pmu.reshape(1,len(pmu))
    ##scatter matrix
    sca_mat = np.zeros((pmu.shape[1],pmu.shape[1]))
    for i in range(data.shape[0]):
        a = data[i,:].reshape(1,pmu.shape[1])
        sca_mat += (a-pmu)*(a-pmu).T
    ##compute eigenvalues and eigenvectors
    w,v = np.linalg.eig(sca_mat)
    pX = np.dot(v.T,data.T)
    ##sort eigenvector by eigenvalues in decreasing order
    orders =np.argsort(w).tolist()
    orders.reverse()
    pX = pX[orders,:]
    ##drop the dimensions and keep error (cumulative variance) under 0.1
    w=np.sort(w)
    errs={}
    for i in range(w.shape[0]):
        errs[(i+1)]=w[:i+1,].sum()/w.sum()
        if w[:i+1,].sum()/w.sum()>0.1:
            pX = pX[:(w.shape[0]-i),:]
            break
    
    return pX.T,v,orders,errs

##data for FLD function has the class label column
def FLD(x,y):
    
    label = np.unique(y)
    ##compute scatter matrix within per class
    sca_w=np.zeros((x.shape[1],x.shape[1]))
    for i in label:
        fmu = np.mean(x[y==i],axis=0)
        fmu = fmu.reshape(1,len(fmu))
        for j in range(x[y==i].shape[0]):
            a = x[y==i][j,:].reshape(1,fmu.shape[1])
            sca_w += (a-fmu)*(a-fmu).T
            
    ##compute scatter matrix between class
    sca_b=np.zeros((x.shape[1],x.shape[1]))
    fmu = np.mean(x,axis=0)
    fmu = fmu.reshape(1,len(fmu))
    for j in range(x.shape[0]):
        a = x[j,:].reshape(1,fmu.shape[1])
        sca_b += (a-fmu)*(a-fmu).T
    sca_b-=sca_w
    ##compute eigenvalues and eigenvectors
    w,v = np.linalg.eig(np.dot(np.linalg.inv(sca_w),sca_b))
    fX = np.dot(v.T,x.T)
    ##select the larget c-1 eigenvalues
    orders =np.argsort(w).tolist()
    orders.reverse()
    fX = fX[orders,:]
    fX = fX[:len(label)-1,:]
    return fX.T,v,orders

##kNN for classification
def kNN(train_x,train_y,test_x,k=5,dist=2):
    
    ##dist is integer and indicates which distance metric is used. when dist=1
    ##it's manhattan distance and when dist=2, it's euclidean distance
    start = timeit.default_timer()
    labels=[]
    prob={}
    for i in range(test_x.shape[0]):
        prob[i]={}
        d=[]
        for j in train_x:
            d.append(np.power((np.power(abs(test_x[i,:]-j),dist)).sum(),1/dist))
        orders =np.argsort(d).tolist()
        y=train_y[orders]
        unic,cout= np.unique(y[:k], return_counts=True)
        unic = unic.tolist()
        cout = cout.tolist()
        for g in range(len(unic)):
            prob[i][unic[g]]=cout[g]/k
        labels.append(max(prob[i].items(), key=operator.itemgetter(1))[0])

    stop = timeit.default_timer()
    print('Time: ', '%05d' % (stop - start),"s")  
    return labels, prob
        
##kNN for classification integrating prior probability and only works for
##two classes
def kNN2(train_x,train_y,test_x,k=5,dist=2,ratio=[0.5,0.5]):
    
    ##dist is integer and indicates which distance metric is used. when dist=1
    ##it's manhattan distance and when dist=2, it's euclidean distance
    #start = timeit.default_timer()
    labels=[]
    prob={}
    for i in range(test_x.shape[0]):
        prob[i]={}
        d=[]
        for j in train_x:
            d.append(np.power((np.power(abs(test_x[i,:]-j),dist)).sum(),1/dist))
        orders =np.argsort(d).tolist()
        y=train_y[orders]
        unic,cout= np.unique(y[:k], return_counts=True)
        unic = unic.tolist()
        cout = cout.tolist()
        if unic[0]==0:
            prob[i][0]=ratio[0]*cout[0]/k
            prob[i][1]=1-prob[i][0]
        else:
            prob[i][1]=ratio[1]*cout[0]/k
            prob[i][0]=1-prob[i][1]
        labels.append(max(prob[i].items(), key=operator.itemgetter(1))[0])

    #stop = timeit.default_timer()
    #print('Time: ', '%05d' % (stop - start),"s")  
    return labels, prob