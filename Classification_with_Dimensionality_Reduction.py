# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:06:33 2019

@author: Yang Xu
"""

##-----------------------------------------------------------------------------
##import modules
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import DimRed as dr

##-----------------------------------------------------------------------------
##import dataset and preprocess
tr = pd.read_table('pima.tr',sep='\s+')
te = pd.read_table('pima.te',sep='\s+')
##convert "Yes" and "No" description to 1 and 0
##in this project, "Yes" indicates having diabetes and "No" indicates not
tr.iloc[:,-1]=pd.factorize(tr.iloc[:,-1],sort=True)[0]
te.iloc[:,-1]=pd.factorize(te.iloc[:,-1],sort=True)[0]
tr=tr.values
te=te.values
##normalization
tr_norm,[mu,std] = dr.normalization(tr[:,:-1])
##normalizing testing set based on training set
te_norm,_= dr.normalization(te[:,:-1],mu,std)

##get prior probability for each class
priors=[] 
label,freq = np.unique(tr[:,-1],return_counts=True)
for i in range(label.shape[0]):
    priors.append(freq[i]/freq.sum())


##-----------------------------------------------------------------------------
##Dimension reduction and performance evaluation
tr_pX,eigvp,orderp,errs = dr.PCA(tr_norm)
te_pX = np.dot(eigvp.T,te_norm.T)
##reorder testing set
te_pX = te_pX[orderp,]
##select the first 5 dimensions
te_pX = te_pX[:5,:]
te_pX = te_pX.T

df_tr = pd.DataFrame(np.column_stack((tr_pX[:,:2],tr[:,-1])),\
                     columns=['PC1', 'PC2', 'class'])
df_te = pd.DataFrame(np.column_stack((te_pX[:,:2],te[:,-1])),\
                     columns=['PC1', 'PC2', 'class'])
##plot training set
sns.lmplot(x="PC1", y="PC2", data=df_tr, fit_reg=False, hue='class', legend=False, \
           palette="Set1")
plt.legend(loc='lower right')
##plot testing set
sns.lmplot(x="PC1", y="PC2", data=df_te, fit_reg=False, hue='class', legend=False, \
           palette="Set1")
plt.legend(loc='lower right')

tr_fX,eigvf,orderf = dr.FLD(tr_norm,tr[:,-1])
te_fX = np.dot(eigvf.T,te_norm.T)
##reorder testing set
te_fX = te_fX[orderf,]
##select the first dimension
te_fX = te_fX[:1,:]
te_fX = te_fX.T

df_tr = pd.DataFrame(np.column_stack((tr_fX,tr[:,-1])),\
                     columns=['FLD1', 'class'])
df_te = pd.DataFrame(np.column_stack((te_fX,te[:,-1])),\
                     columns=['FLD1', 'class'])
##plot training set
sns.kdeplot(tr_fX[:,0][tr[:,-1]==0], label='0', shade=True)
sns.kdeplot(tr_fX[:,0][tr[:,-1]==1], label='1', shade=True)
plt.xlabel("FLD1")
##plot testing set
sns.kdeplot(te_fX[:,0][te[:,-1]==0], label='0', shade=True)
sns.kdeplot(te_fX[:,0][te[:,-1]==1], label='1', shade=True)
plt.xlabel("FLD1")

##-----------------------------------------------------------------------------
##Classification by using normalized dataset
##parameter estimation for 3 cases
para_case1 = dr.para_est(tr_norm,tr[:,-1],metrics="case1")
para_case2 = dr.para_est(tr_norm,tr[:,-1],metrics="case2")
para_case3 = dr.para_est(tr_norm,tr[:,-1],metrics="case3")
case1_pred=[]
case2_pred=[]
case3_pred=[]
for i in te_norm:
    prediction,_=dr.bayes_des_rule(para=para_case1,test=i,prior=priors)
    case1_pred.append(prediction)
    prediction,_=dr.bayes_des_rule(para=para_case2,test=i,prior=priors)
    case2_pred.append(prediction)
    prediction,_=dr.bayes_des_rule(para=para_case3,test=i,prior=priors)
    case3_pred.append(prediction)

##accuracy of kNN using nX 
k_acu=[]            
for i in range(1,16):
    knn_pred, _=dr.kNN(tr_norm,tr[:,-1],te_norm,k=i,prior=priors)
    k_acu.append(accuracy_score(te[:,-1], knn_pred))
plt.plot(range(1,16),k_acu)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()

##confusion matrix of case1 using nX 
cm_case1 = confusion_matrix(te[:,-1], case1_pred)
##confusion matrix of case2 using nX 
cm_case2 = confusion_matrix(te[:,-1], case2_pred)
##confusion matrix of case3 using nX 
cm_case3 = confusion_matrix(te[:,-1], case3_pred)
##confusion matrix of knn using nX (k=1) 
knn_pred, _=dr.kNN(tr_norm,tr[:,-1],te_norm,k=1,prior=priors)
cm_knn = confusion_matrix(te[:,-1], knn_pred)

##varing prior probability and plot ROC curve
label,freq = np.unique(te[:,-1],return_counts=True)
prior0=np.linspace(0.15, 0.85)
prior0=prior0.tolist()
sen_spec=np.empty(shape=(0,8))
rocs=np.empty(shape=(0,8))
s=freq[1]
for i in prior0:
    prior_p = [1-i,i]
    case1_pred=[]
    case2_pred=[]
    case3_pred=[]
    
    for j in te_norm:
        prediction,_=dr.bayes_des_rule(para=para_case1,test=j,prior=prior_p)
        case1_pred.append(prediction)
        prediction,_=dr.bayes_des_rule(para=para_case2,test=j,prior=prior_p)
        case2_pred.append(prediction)
        prediction,_=dr.bayes_des_rule(para=para_case3,test=j,prior=prior_p)
        case3_pred.append(prediction)
    knn_pred, _=dr.kNN(tr_norm,tr[:,-1],te_norm,k=1,prior=prior_p)
    cm_knn = confusion_matrix(te[:,-1], knn_pred)
    cm_case1 = confusion_matrix(te[:,-1], case1_pred)
    cm_case2 = confusion_matrix(te[:,-1], case2_pred)
    cm_case3 = confusion_matrix(te[:,-1], case3_pred)
    
    rocs=np.vstack((rocs,np.array((cm_case1[1][0]/s,cm_case1[1][1]/s,\
                                   cm_case2[1][0]/s,cm_case2[1][1]/s,\
                                   cm_case3[1][0]/s,cm_case3[1][1]/s,\
                                   cm_knn[1][0]/s,cm_knn[1][1]/s))))
    sen_spec=np.vstack((sen_spec,np.array((cm_case1[1][1]/(cm_case1[1][1]+cm_case1[1][0]),\
                                      cm_case1[0][0]/(cm_case1[0][0]+cm_case1[0][1]),\
                                      cm_case2[1][1]/(cm_case2[1][1]+cm_case2[1][0]),\
                                      cm_case2[0][0]/(cm_case2[0][0]+cm_case2[0][1]),\
                                      cm_case3[1][1]/(cm_case3[1][1]+cm_case3[1][0]),\
                                      cm_case3[0][0]/(cm_case3[0][0]+cm_case3[0][1]),\
                                      cm_knn[1][1]/(cm_knn[1][1]+cm_knn[1][0]),\
                                      cm_knn[0][0]/(cm_knn[0][0]+cm_knn[0][1])))))
    
##plot sensitivity
plt.plot(prior0, sen_spec[:,0], color='red', label="case1")
plt.plot(prior0, sen_spec[:,2], color='yellow', label="case2")
plt.plot(prior0, sen_spec[:,4], color='green', label="case3")
plt.plot(prior0, sen_spec[:,6], color='black', label="kNN (k=1)")
plt.xlabel("prior probability of class 1")
plt.ylabel("sensitivity")
plt.legend()
##plot specificity
plt.plot(prior0, sen_spec[:,1], color='red', label="case1")
plt.plot(prior0, sen_spec[:,3], color='yellow', label="case2")
plt.plot(prior0, sen_spec[:,5], color='green', label="case3")
plt.plot(prior0, sen_spec[:,7], color='black', label="kNN (k=1)")
plt.xlabel("prior probability of class 1")
plt.ylabel("specificity")
plt.legend()

##plot roc curve
plt.plot(rocs[:,0], rocs[:,1], color='red', label="case1")
plt.plot(rocs[:,2], rocs[:,3], color='yellow', label="case2")
plt.plot(rocs[:,4], rocs[:,5], color='green', label="case3")
plt.plot(rocs[:,6], rocs[:,7], color='black', label="kNN (k=1)")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend()


##-----------------------------------------------------------------------------
##Classification by using dimension reduced dataset
##PCA
para_case1 = dr.para_est(tr_pX,tr[:,-1],metrics="case1")
para_case2 = dr.para_est(tr_pX,tr[:,-1],metrics="case2")
para_case3 = dr.para_est(tr_pX,tr[:,-1],metrics="case3")
case1_pred=[]
case2_pred=[]
case3_pred=[]
for i in te_pX:
    prediction,_=dr.bayes_des_rule(para=para_case1,test=i,prior=priors)
    case1_pred.append(prediction)
    prediction,_=dr.bayes_des_rule(para=para_case2,test=i,prior=priors)
    case2_pred.append(prediction)
    prediction,_=dr.bayes_des_rule(para=para_case3,test=i,prior=priors)
    case3_pred.append(prediction)

##accuracy of kNN using nX 
k_acu=[]            
for i in range(1,16):
    knn_pred, _=dr.kNN(tr_pX,tr[:,-1],te_pX,k=i,prior=priors)
    k_acu.append(accuracy_score(te[:,-1], knn_pred))
plt.plot(range(1,16),k_acu)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()

##confusion matrix of case1 using nX 
cm_case1 = confusion_matrix(te[:,-1], case1_pred)
##confusion matrix of case2 using nX 
cm_case2 = confusion_matrix(te[:,-1], case2_pred)
##confusion matrix of case3 using nX 
cm_case3 = confusion_matrix(te[:,-1], case3_pred)
##confusion matrix of knn using nX (k=1) 
knn_pred, _=dr.kNN(tr_pX,tr[:,-1],te_pX,k=1,prior=priors)
cm_knn = confusion_matrix(te[:,-1], knn_pred)

##varing prior probability and plot ROC curve
label,freq = np.unique(te[:,-1],return_counts=True)
prior0=np.linspace(0.15, 0.85)
prior0=prior0.tolist()
sen_spec=np.empty(shape=(0,8))
rocs=np.empty(shape=(0,8))
s=freq[1]
for i in prior0:
    prior_p = [1-i,i]
    case1_pred=[]
    case2_pred=[]
    case3_pred=[]
    
    for j in te_pX:
        prediction,_=dr.bayes_des_rule(para=para_case1,test=j,prior=prior_p)
        case1_pred.append(prediction)
        prediction,_=dr.bayes_des_rule(para=para_case2,test=j,prior=prior_p)
        case2_pred.append(prediction)
        prediction,_=dr.bayes_des_rule(para=para_case3,test=j,prior=prior_p)
        case3_pred.append(prediction)
    knn_pred, _=dr.kNN(tr_pX,tr[:,-1],te_pX,k=1,prior=prior_p)
    cm_knn = confusion_matrix(te[:,-1], knn_pred)
    cm_case1 = confusion_matrix(te[:,-1], case1_pred)
    cm_case2 = confusion_matrix(te[:,-1], case2_pred)
    cm_case3 = confusion_matrix(te[:,-1], case3_pred)
    
    rocs=np.vstack((rocs,np.array((cm_case1[1][0]/s,cm_case1[1][1]/s,\
                                   cm_case2[1][0]/s,cm_case2[1][1]/s,\
                                   cm_case3[1][0]/s,cm_case3[1][1]/s,\
                                   cm_knn[1][0]/s,cm_knn[1][1]/s))))
    sen_spec=np.vstack((sen_spec,np.array((cm_case1[1][1]/(cm_case1[1][1]+cm_case1[1][0]),\
                                      cm_case1[0][0]/(cm_case1[0][0]+cm_case1[0][1]),\
                                      cm_case2[1][1]/(cm_case2[1][1]+cm_case2[1][0]),\
                                      cm_case2[0][0]/(cm_case2[0][0]+cm_case2[0][1]),\
                                      cm_case3[1][1]/(cm_case3[1][1]+cm_case3[1][0]),\
                                      cm_case3[0][0]/(cm_case3[0][0]+cm_case3[0][1]),\
                                      cm_knn[1][1]/(cm_knn[1][1]+cm_knn[1][0]),\
                                      cm_knn[0][0]/(cm_knn[0][0]+cm_knn[0][1])))))
    
##plot sensitivity
plt.plot(prior0, sen_spec[:,0], color='red', label="case1")
plt.plot(prior0, sen_spec[:,2], color='yellow', label="case2")
plt.plot(prior0, sen_spec[:,4], color='green', label="case3")
plt.plot(prior0, sen_spec[:,6], color='black', label="kNN (k=1)")
plt.xlabel("prior probability of class 1")
plt.ylabel("sensitivity")
plt.legend()
##plot specificity
plt.plot(prior0, sen_spec[:,1], color='red', label="case1")
plt.plot(prior0, sen_spec[:,3], color='yellow', label="case2")
plt.plot(prior0, sen_spec[:,5], color='green', label="case3")
plt.plot(prior0, sen_spec[:,7], color='black', label="kNN (k=1)")
plt.xlabel("prior probability of class 1")
plt.ylabel("specificity")
plt.legend()

##plot roc curve
plt.plot(rocs[:,0], rocs[:,1], color='red', label="case1")
plt.plot(rocs[:,2], rocs[:,3], color='yellow', label="case2")
plt.plot(rocs[:,4], rocs[:,5], color='green', label="case3")
plt.plot(rocs[:,6], rocs[:,7], color='black', label="kNN (k=1)")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend()

##FLD
para_case1 = dr.para_est(tr_fX,tr[:,-1],metrics="case1")
para_case2 = dr.para_est(tr_fX,tr[:,-1],metrics="case2")
para_case3 = dr.para_est(tr_fX,tr[:,-1],metrics="case3")
case1_pred=[]
case2_pred=[]
case3_pred=[]
for i in te_fX:
    prediction,_=dr.bayes_des_rule(para=para_case1,test=i,prior=priors)
    case1_pred.append(prediction)
    prediction,_=dr.bayes_des_rule(para=para_case2,test=i,prior=priors)
    case2_pred.append(prediction)
    prediction,_=dr.bayes_des_rule(para=para_case3,test=i,prior=priors)
    case3_pred.append(prediction)

##accuracy of kNN using nX 
k_acu=[]            
for i in range(1,16):
    knn_pred, _=dr.kNN(tr_fX,tr[:,-1],te_fX,k=i,prior=priors)
    k_acu.append(accuracy_score(te[:,-1], knn_pred))
plt.plot(range(1,16),k_acu)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()

##confusion matrix of case1 using nX 
cm_case1 = confusion_matrix(te[:,-1], case1_pred)
##confusion matrix of case2 using nX 
cm_case2 = confusion_matrix(te[:,-1], case2_pred)
##confusion matrix of case3 using nX 
cm_case3 = confusion_matrix(te[:,-1], case3_pred)
##confusion matrix of knn using nX (k=1) 
knn_pred, _=dr.kNN(tr_fX,tr[:,-1],te_fX,k=1,prior=priors)
cm_knn = confusion_matrix(te[:,-1], knn_pred)

##varing prior probability and plot ROC curve
label,freq = np.unique(te[:,-1],return_counts=True)
prior0=np.linspace(0.15, 0.85)
prior0=prior0.tolist()
sen_spec=np.empty(shape=(0,8))
rocs=np.empty(shape=(0,8))
s=freq[1]
for i in prior0:
    prior_p = [1-i,i]
    case1_pred=[]
    case2_pred=[]
    case3_pred=[]
    
    for j in te_fX:
        prediction,_=dr.bayes_des_rule(para=para_case1,test=j,prior=prior_p)
        case1_pred.append(prediction)
        prediction,_=dr.bayes_des_rule(para=para_case2,test=j,prior=prior_p)
        case2_pred.append(prediction)
        prediction,_=dr.bayes_des_rule(para=para_case3,test=j,prior=prior_p)
        case3_pred.append(prediction)
    knn_pred, _=dr.kNN(tr_fX,tr[:,-1],te_fX,k=1,prior=prior_p)
    cm_knn = confusion_matrix(te[:,-1], knn_pred)
    cm_case1 = confusion_matrix(te[:,-1], case1_pred)
    cm_case2 = confusion_matrix(te[:,-1], case2_pred)
    cm_case3 = confusion_matrix(te[:,-1], case3_pred)
    
    rocs=np.vstack((rocs,np.array((cm_case1[1][0]/s,cm_case1[1][1]/s,\
                                   cm_case2[1][0]/s,cm_case2[1][1]/s,\
                                   cm_case3[1][0]/s,cm_case3[1][1]/s,\
                                   cm_knn[1][0]/s,cm_knn[1][1]/s))))
    sen_spec=np.vstack((sen_spec,np.array((cm_case1[1][1]/(cm_case1[1][1]+cm_case1[1][0]),\
                                      cm_case1[0][0]/(cm_case1[0][0]+cm_case1[0][1]),\
                                      cm_case2[1][1]/(cm_case2[1][1]+cm_case2[1][0]),\
                                      cm_case2[0][0]/(cm_case2[0][0]+cm_case2[0][1]),\
                                      cm_case3[1][1]/(cm_case3[1][1]+cm_case3[1][0]),\
                                      cm_case3[0][0]/(cm_case3[0][0]+cm_case3[0][1]),\
                                      cm_knn[1][1]/(cm_knn[1][1]+cm_knn[1][0]),\
                                      cm_knn[0][0]/(cm_knn[0][0]+cm_knn[0][1])))))
    
##plot sensitivity
plt.plot(prior0, sen_spec[:,0], color='red', label="case1")
plt.plot(prior0, sen_spec[:,2], color='yellow', label="case2")
plt.plot(prior0, sen_spec[:,4], color='green', label="case3")
plt.plot(prior0, sen_spec[:,6], color='black', label="kNN (k=1)")
plt.xlabel("prior probability of class 1")
plt.ylabel("sensitivity")
plt.legend()
##plot specificity
plt.plot(prior0, sen_spec[:,1], color='red', label="case1")
plt.plot(prior0, sen_spec[:,3], color='yellow', label="case2")
plt.plot(prior0, sen_spec[:,5], color='green', label="case3")
plt.plot(prior0, sen_spec[:,7], color='black', label="kNN (k=1)")
plt.xlabel("prior probability of class 1")
plt.ylabel("specificity")
plt.legend()

##plot roc curve
plt.plot(rocs[:,0], rocs[:,1], color='red', label="case1")
plt.plot(rocs[:,2], rocs[:,3], color='yellow', label="case2")
plt.plot(rocs[:,4], rocs[:,5], color='green', label="case3")
plt.plot(rocs[:,6], rocs[:,7], color='black', label="kNN (k=1)")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend()
