#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:30:39 2017

@author: johnklein
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_circles, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle
from scipy.stats import beta, norm
import sys


def my_make_moons(n_samples=100, shuffle=True, noise=None, random_state=None):
    """This is a slightly modified function as compared to the sklearn version.
    This functions makes two interleaving half circles
    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`. 
    In this version, the position of sample points on the half circles is also
    random.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    generator = check_random_state(random_state)

    angle = generator.uniform(size=(n_samples_out,))*np.pi
    outer_circ_x = np.cos(angle)
    outer_circ_y = np.sin(angle)
    angle = generator.uniform(size=(n_samples_out,))*np.pi    
    inner_circ_x = 1 - np.cos(angle)
    inner_circ_y = 1 - np.sin(angle) - .5

    X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise != None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y


def kfold_data(X,c,k,n_class):
    """
    A function to split a dataset according to the statified k fold cross validation
    principle. If there is less class representative example than folds, some
    folds may not contain any example of this class.
    """
    (d,n)=X.shape
    X_test_folds=[]
    c_test_folds=[]
    X_train_folds=[]
    c_train_folds=[]
    myrows=np.array(range(d),dtype=np.int)
    for i in range(k):
        for j in range(n_class):
            mycols=np.where(c==j)
            X_loc=X[myrows[:, np.newaxis], mycols]
            c_loc=c[mycols]
            (d,n_loc)=X_loc.shape
            fold_size=int(n_loc/k)
            if (fold_size>0):
                mycols=np.array(range(i*fold_size,(i+1)*fold_size),dtype=np.int)
            else:
                if (i<len(c_loc)):
                    mycols=np.array([i],dtype=np.int)
                else:
                    mycols=np.array([],dtype=np.int)
            if (j==0):
                X_test_fold=X_loc[myrows[:, np.newaxis], mycols]
                c_test_fold=c_loc[mycols]
            else :
                X_test_fold=np.hstack((X_test_fold,X_loc[myrows[:, np.newaxis], mycols]))
                c_test_fold=np.hstack((c_test_fold,c_loc[mycols]))
            mycols=np.setdiff1d(np.arange(0,n_loc), mycols)
            if (j==0):
                X_train_fold=X_loc[myrows[:, np.newaxis], mycols]
                c_train_fold=c_loc[mycols]
            else :
                X_train_fold=np.hstack((X_train_fold,X_loc[myrows[:, np.newaxis], mycols]))
                c_train_fold=np.hstack((c_train_fold,c_loc[mycols]))
        X_test_folds.append(X_test_fold)
        c_test_folds.append(c_test_fold)
        X_train_folds.append(X_train_fold)
        c_train_folds.append(c_train_fold)        
    return X_train_folds, c_train_folds, X_test_folds, c_test_folds

def indep(cond_cdf,n_class):
    """
    A function to compute the conditional joint cdfs from conditional maginal 
    ones using to the indepedent copula model.
    The joint cdf is 2D, meaning that it applies to a pair of clfs.
    The function cannot be used if the number of classes is too large.
    """
    joint_cdf = np.zeros((n_class,n_class,n_class))
    for i in range(n_class):
        for j in range(n_class):
            joint_cdf[j,:,i] = cond_cdf[0][j,i]*cond_cdf[1][:,i]
    return joint_cdf


def pdf2cdf(pdf):
    """
    This function turns a set of joint pdfs to a set of joint cdfs.
    Warning: only works for a pair of classifiers.
    input pdf is a collection of conditional 2D joint cdfs (given true class).
    """
    (Ns_loc,n_class1,n_class2) = pdf.shape
    if (n_class1 != n_class2):
        raise ValueError('The second and third dimensions must agree (number of classes).')
    cdf = np.zeros((Ns_loc,n_class1,n_class1))
    for i in range(Ns_loc):
        for j in range(n_class1):
            cdf[i][:,j] = np.cumsum(pdf[i][:,j])
    return cdf


def gausscop_pdf(u,theta):
    """
    Computes the combined predictions based on those returned by classifiers.
    
    Parameters
    ----------    
     - u : numpy array containing the conditional cdf of each classifier 
    - theta : the classifier covariance parameter. This parameter lives in the 
    open interval (-1/(len(u)-1);1).
    
    Returns
    -------
    - cop : scalar (copula value)
    """
    invmat = 1/(1-theta) * (np.eye(len(u)) - theta/(1+(len(u)-1)*theta)*np.ones((len(u),len(u))) )
    det = (1+(len(u)-1)*theta)*np.power(1-theta,len(u)-1)
    precision = sys.float_info.epsilon *1e6
    if (len(u[np.where(u>precision)])>0):
        mini = np.min(u[np.where(u>0)])*1e-6 #something a lot smaller than the minimal positive value in u
        mini = max(mini,precision)
    else: # all entries in u are null or negative
        mini = precision
    if (len(u[np.where(u<1-precision)])>0):
        maxi = 1-(1-np.max(u[np.where(u<1)]))*1e-6 #something a lot larger than the maximal value in u that is below 1
        maxi = min(maxi,1-precision)
    else: # all entries in u are 1 or something bigger
        maxi = 1 - precision
    u[np.where(u<=0)] = mini #avoid -inf
    u[np.where(u>=1)] = maxi #avoid +inf     
    if (np.sum(u>=1)):
        print(u)
        raise ValueError('stop')
    v = norm.ppf(u)
    
    #cop = 1/np.sqrt(det) * np.exp(np.dot(-0.5*v, np.dot(invmat-np.eye(len(u)),v) ))
    log_cop = np.dot(-0.5*v, np.dot(invmat-np.eye(len(u)),v) ) - 0.5*np.log(det)
    return log_cop    

def combinedPred(preds,cond_pdfs,priors,copula='indep',theta=None,cond_cdfs=None):
    """
    Computes the combined predictions based on those returned by classifiers.
    
    Parameters
    ----------    
     - preds : a 2D numpy array of int
         It containing the predictions returned by each 
         classifier individually. The 1st dimension is the number of classifier.
         The second dimension is the number of examples from which the predictions 
         are obtained.
    
    - cond_pdfs : 3D numpy array of float
        It is a collection of conditional pdfs of predicted class given true 
        class for each classifier. The 1st dimension is the classifier index.
        The second dimension is the predicted class index. The third dimension
        is the true class index
        
    - priors : 1D numpy array of float
        Probabilities of true classes.
        
    - copula : string
        The name of the chosen copula function (indep,clayton or franck)
        
    - theta : float or 1D numpy array of float
        The parameter of the copula function (only if copula is not indep).
        The parameters can be different for each true class.
    Returns
    -------
    - pred_out : 1D numpy array
        This array contains the prediction of the classifier ensemble for each
        example.
    """
    n_class = priors.shape[0]
    if (preds.ndim==1):
        Ns_loc = preds.size
        n_loc = 1
        my_preds = np.reshape(preds,(Ns_loc,1))
    if (preds.ndim==2):
        (Ns_loc,n_loc) = preds.shape
        my_preds = preds
    if (np.isscalar(theta) == True):
        my_theta = theta*np.ones((n_class,))
    else:
        my_theta = theta
    (Ns_loc,n_class1,n_class2) = cond_pdfs.shape
    pred_out = np.zeros((n_loc,),dtype=int)
    if (n_class1 != n_class2):
        raise ValueError('The second and third dimensions must agree (number of classes).')    
    for i in range(n_loc):#loop on examples
        pdf = np.asarray([])
        for j in range(Ns_loc):     
            pdf = np.append(pdf,cond_pdfs[j,my_preds[j,i],:])
        pdf = np.reshape(pdf,(Ns_loc,n_class1))
        comb_pdf = np.prod(pdf,axis=0)
        #if (my_preds[2,i]!=my_preds[1,i]):
            #print(pdf,comb_pdf)
        if (copula != 'indep'):
            cdf = np.asarray([])
            for j in range(Ns_loc):     
                cdf = np.append(cdf,cond_cdfs[j,my_preds[j,i],:])
            cdf = np.reshape(cdf,(Ns_loc,n_class1))
            cop = np.zeros((n_class1,))
            for j in range(n_class1):
                if (copula == 'Gaussian'):
                    cop[j] = gausscop_pdf(cdf[:,j],my_theta[j])
            if (copula == 'Gaussian'):
                comb_pdf = np.log(comb_pdf) + cop
            else:
                comb_pdf = np.multiply(comb_pdf,cop)
        if (copula == 'Gaussian'):
            pred_out[i] = np.argmax(np.log(priors) + comb_pdf)
        else:
            pred_out[i] = np.argmax(np.multiply(comb_pdf,priors))
    return pred_out

def gen_blobs(n):
    """
    This function generates a dataset from the following generating process: 
    4 gaussian 2D distributions centered on each corner of a centered square
    whose side length is 4. Each gaussian has unit variance
    The diagonal blobs generates example of class n° 0. The anti-diagonal blobs 
    generates examples of class n°1 and n°2 respectively.
    """
    X, y = make_blobs(n_samples=int(0.75*n),n_features=2, centers=np.asarray([[-2,-2],[-2,2],[2,-2] ]))
    X2, y2 = make_blobs(n_samples=int(0.25*n),n_features=2, centers=np.asarray([[2,2] ]))
    X = np.vstack((X,X2))
    y = np.hstack((y,y2))
    return (X,y)

def weighted_vote(preds,acc,n_class):
    """
    A simple classifier combination based on a weighted vote. Each classifier 
    vote is weighted according to its estimated accuraccy.
    """
    (Ns,n) = preds.shape
    votes = np.zeros((n,n_class))
    for i in range(Ns):
        votes[range(n), preds[i,:]] += acc[i]
    pred_out = np.argmax(votes,axis=1)
    return pred_out
    
def opt_pred(name,X):
    """
    Optimal classifiers for each generating process.
    """
    preds = np.zeros((X.shape[0],),dtype=int)
    if (name == 'blobs'):
        ind = np.where((X[:,0]<0) & (X[:,1]>0))
        preds[ind] = 1
        ind = np.where((X[:,0]>0) & (X[:,1]<0))
        preds[ind] = 2
    elif (name == 'moons'):
        angle = np.linspace(0,np.pi,100)
        outer_circ_x = np.cos(angle)
        outer_circ_y = np.sin(angle)
        inner_circ_x = 1 - np.cos(angle)
        inner_circ_y = 1 - np.sin(angle) - .5        
        for i in range(X.shape[0]):
            d0 = (outer_circ_x-X[i,:][0])**2 + (outer_circ_y-X[i,:][1])**2
            p0 = np.sum(np.exp(-d0/(2*0.3**2)))
            d1 = (inner_circ_x-X[i,:][0])**2 + (inner_circ_y-X[i,:][1])**2
            p1 = np.sum(np.exp(-d1/(2*0.3**2)))
            if (p1>p0):
                preds[i]=1
    elif (name == 'circles'):
        d = (X[:,0])**2 + (X[:,1])**2
        preds = (d<0.75**2).astype(int)
    else:
        raise ValueError('Unknown generating process name.')
    return preds

def get_local_data(X,y,i,name):
    """
    This functions returns the piece of the dataset that will be used to train
    classifier number i.
    """
    if (name == 'blobs'):
        if (i==0):
            ind = np.where(X[:,0]<-X[:,1])[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==1):
            ind = np.where(X[:,0]>=-X[:,1])[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]                
    elif (name == 'moons'):
        if (i==0):
            ind = np.where(X[:,0]<0)[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==1):
            ind = np.where((X[:,0]>=0) & (X[:,0]<1))[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==2):
            ind = np.where(X[:,0]>=1)[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]            
    elif (name == 'circles'):
        theta = np.arctan2(X[:,1], X[:,0])
        if (i==0):
            ind = np.where(theta<-np.pi/3)[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==1):
            ind = np.where((theta>=-np.pi/3) & (theta<np.pi/3))[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]
        if (i==2):
            ind = np.where(theta>=np.pi/3)[0]
            X_train_loc = X[ind,:]
            y_train_loc = y[ind]            
    else:
        raise ValueError('Unknown generating process name.')
    return X_train_loc,y_train_loc

def get_split_data(X_all,y_all,name,mode,Ns,n_class):
    """
    This function splits the dataset into several pieces - on for each classifier
    to be trained. The splitting scheme is different for each generating process.
    """
    X=[]
    y=[]
    if (mode == 'deterministic_split'):
        for i in range(Ns):
            (X_loc,y_loc) = get_local_data(X_all,y_all,i,name)
            X.append(X_loc)
            y.append(y_loc)
    elif (mode == 'random_split'):
        X_train_folds, y_train_folds, X_test_folds, y_test_folds = kfold_data(X_all.T,y_all,Ns,n_class)
        for i in range(Ns):
            X_loc = X_test_folds[i].T
            y_loc = y_test_folds[i]
            X.append(X_loc)
            y.append(y_loc)
    else:
        raise ValueError('Unknown spliting mode name.')
    return X,y

def normalize(x):
    """
    This functions normalizes a numpy array (x) so that it sums to one.
    """
    return x/np.sum(x)

def simple_split(X,y,percent,n_class):
    """
    This function splits the data in two subsets. The fraction of the data kept
    by the 1st subset is specified by the parameter "percent".
    """
    cards = np.zeros((n_class,))
    inds = []
    for i in range(np.min(y),np.min(y)+n_class):
        cards[i-np.min(y)] = np.sum(y==i)
        proportion = int(percent*cards[i-np.min(y)])
        if (proportion == 0):
            proportion = 1
        inds = inds + list(np.where(y==i)[0][:proportion]) 
    X1 = X[inds]
    y1 = y[inds]
    inds2 = np.setdiff1d(range(len(y)),inds)
    X2 = X[inds2]
    y2 = y[inds2]   
    return X1,y1,X2,y2

def launch_test(dataset,mode,n,iter_max=1e6,sent=0.1,copula='Gaussian'):
    """
    This functions launches the DELCO ensemble method as well as other reference
    methods.
    
    Parameters
    ----------    
    - dataset : string
        It specifies the name of the synthetic dataset on which methods are tested.
        Possible choices are 'moons', 'circles' and 'blobs'.
    
    - mode : string
        It specifies the type of splits for local nodes (base classifiers). To 
        reproduce the paper's results choose 'deterministic_split'. Another 
        available type of split is 'random_split'.
        
    - n : int
        dataset size.
        
    - iter_max : int
        Maximal number of loops. It should be set to np.inf to achieve the 
        prescribed confidence level.
        
    - sent: float
        Percentage of the data sent to the central node.
    
    - copula: string
        The name of the chosen copula for DELCO. Possible choices are 'Gaussian'
        and 'indep'
    
    - 
    Returns
    -------
    none. Accuracies of the methods are printed.    
    """
    start = time.time()
    ##########################################
    # PARAMETERS BELOW SHOULD NOT BE CHANGED #
    ##########################################
    epsi = 1e-7
    #classifiers (decentralized logistic regressors)
    if (dataset == 'blobs'):
            names = [ "Reg Log","Reg Log"]
    else:
            names = [ "Reg Log","Reg Log","Reg Log"]    
 
    Ns = len(names)
 
    #base classifier instances
    clf = []
    for i in range(Ns):
        if names[i] == 'Reg Log':    
            clf.append(LogisticRegression(penalty='l2', C=1.0))
            
    #central classifier instance
    clf_ctl = LogisticRegression(penalty='l2', C=1.0)
    
    #second level classifier instance (for stacking)
    clf_meta = LogisticRegression(penalty='l2', C=1.0)
            
    if (copula != 'indep'):
        if (copula == 'Gaussian'):
            theta_range = np.linspace(-1.0/(Ns-1)+1e-2,0.99,101)
        else:
            raise ValueError('Unknown copula name.')
    
    degenerate = True
    while (degenerate == True):
        if (dataset == 'blobs'):
            (X_all,y_all) = gen_blobs(n)
        elif (dataset == 'moons'):
            (X_all,y_all) = my_make_moons(n_samples=n,noise=0.3)
        elif (dataset == 'circles'):
            (X_all,y_all) = make_circles(n_samples=n, factor=.5, noise=.15)
        else:
            raise ValueError('Unknown generating process name.')

        n_class = y_all.max()-y_all.min()+1
                    
        #Spliting data for each local clf
        (X_loc,y_loc) = get_split_data(X_all,y_all,dataset,mode,Ns,n_class)
        
        #Selecting the send examples to central machine for those methods with budgeted training
        X_loc_small = []
        y_loc_small = []
        for i in range(Ns):
            X_s, y_s, X_i, y_i = simple_split(X_loc[i],y_loc[i],sent,n_class)
            X_loc_small.append(X_i)
            y_loc_small.append(y_i)
            if (i==0):
                X_sent = X_s
                y_sent = y_s                  
            else:
                X_sent = np.vstack((X_sent, X_s))
                y_sent = np.hstack((y_sent, y_s))
        
        degenerate = False      
        for i in range(Ns):
            for j in range(n_class):
                if (np.sum(y_loc_small[i]==j)<1):
                    degenerate = True                    
                if (np.sum(y_sent==j)==0):
                    degenerate = True
    print('dataset ok')
 
    #Estimation of conditional probabilities (predicted class given actual class)
    cond_pdf = np.zeros((Ns,n_class,n_class))
    score_clf = np.zeros((Ns,))
    preds = np.zeros((Ns,y_sent.size),dtype=int)
    for i in range(Ns):#Classifier number
        #Training clfs on decentralized machines on datapoints that are not sent to central machine
        clf[i].fit(X_loc_small[i],y_loc_small[i])
        #Evaluation in the central machine
        score_clf[i] = clf[i].score(X_sent,y_sent)
        preds[i] = clf[i].predict(X_sent)
        for j in range(n_class):#True class index
            ind = np.where(y_sent==j)[0]
            for k in range(n_class):#Predicted class index
                cond_pdf[i][k,j] += (np.sum(preds[i,ind]==k)+1)/(ind.size+n_class) #Laplace add one
    
    if(degenerate == False):#Going on only if it is worth
        select = np.argmax(score_clf)
        
        #Conditional cumulative distributions
        cond_cdf=pdf2cdf(cond_pdf)
        
        #True class probabilities (in central machine)
        prior = np.zeros((n_class,))
        for i in range(n_class):
            prior[i] = np.sum(y_sent==i)/y_sent.size
        
        #Stacked log reg training
        clf_meta.fit(preds.T,y_sent)
        
        #Copula parameter estimation (grid search) using the sent data on the central machine
        if (copula != 'indep'):
            theta = np.zeros((n_class,))
            theta_success = np.zeros((theta_range.size,))
            #For each (centralized) test data
            for i in range(y_sent.size):
                #Starting the computatuon of the copula based combination prediction
                pdf_loc = np.asarray([])
                cdf_loc = np.asarray([])
                for k in range(Ns):     
                    pdf_loc = np.append(pdf_loc,cond_pdf[k,preds[k,i],:])
                    cdf_loc = np.append(cdf_loc,cond_cdf[k,preds[k,i],:])
                pdf_loc = np.reshape(pdf_loc,(Ns,n_class))
                cdf_loc = np.reshape(cdf_loc,(Ns,n_class))
                comb_pdf = np.prod(pdf_loc,axis=0)
                #Loop on values of parameter theta
                for j in range(theta_range.size):
                    cop = np.zeros((n_class,))
                    for k in range(n_class):
                        if (copula == 'Gaussian'):
                            cop[k] = gausscop_pdf(cdf_loc[:,k],theta_range[j])
                    if (copula == 'Gaussian'):
                        joint = np.log(comb_pdf) + cop
                        posterior = np.log(prior) + joint
                    else:                            
                        joint = np.multiply(comb_pdf,cop)
                        posterior = np.multiply(joint,prior)
                    comb_pred = np.argmax(posterior)
                    if (comb_pred == y_sent[i]):
                        theta_success[j] += 1
            theta_success = theta_success/y_sent.size
            theta = np.median(theta_range[np.where(theta_success == np.max(theta_success))])
        
        #Retrain base classifiers on the whole training set
        for i in range(Ns):
            clf[i].fit(X_loc[i],y_loc[i]) 
            
        #Traning in centralized fashion
        clf_ctl.fit(X_all,y_all)            
            
        #Start testing on newly generated examples
        n_test = 0
        n_test_batch = 10000
        n_success = 0
        n_success_select = 0
        n_success_opt = 0
        n_success_ind = 0
        n_success_vote = 0
        n_success_ctl = 0
        n_success_met = 0
        clf_rates = np.zeros((Ns,))
        alpha = 0.05
        clopper_pearson_interval = np.ones((Ns+7))
        iter_nb = 0
        print('entering main loop')
        #Looping until Clopper Pearson interval meets prescribed conditions for the accurracy of each prediction method
        while ((np.max(clopper_pearson_interval) > 0.002) and (iter_nb<iter_max)): #3e5
            if (dataset == 'blobs'):
                (X_test,y_test) = gen_blobs(n_test_batch)
            elif (dataset == 'moons'):
                (X_test,y_test)  = my_make_moons(n_samples=n_test_batch,noise=0.3)
            elif (dataset == 'circles'):
                (X_test,y_test) = make_circles(n_samples=n_test_batch, factor=.5, noise=.15)
            else:
                raise ValueError('Unknown generating process name.')            
            n_test += y_test.size
            preds = np.zeros((Ns,y_test.size),dtype=int)
            for i in range(Ns):
                preds[i,:] = clf[i].predict(X_test)
            combined_pred = np.zeros((y_test.size,))
            for i in range(y_test.size):
                combined_pred[i] = combinedPred(preds[:,i],cond_pdf,prior,copula=copula,theta=theta,cond_cdfs=cond_cdf)
            n_success += np.sum(combined_pred==y_test)
            for i in range(Ns):
                clf_rates[i] += np.sum(preds[i,:]==y_test) 
                clopper_pearson_interval[i] = beta.ppf(1-alpha/2,clf_rates[i]+1,n_test-clf_rates[i]+epsi) - beta.ppf(alpha/2,clf_rates[i]+epsi,n_test-clf_rates[i]+1)
            n_success_select += np.sum(preds[select,:]==y_test)
            #clf selection 
            clopper_pearson_interval[Ns] = beta.ppf(1-alpha/2,n_success_select+1,n_test-n_success_select+epsi) - beta.ppf(alpha/2,n_success_select+epsi,n_test-n_success_select+1)
            #clf combination
            clopper_pearson_interval[Ns+1] = beta.ppf(1-alpha/2,n_success+1,n_test-n_success+epsi) - beta.ppf(alpha/2,n_success+epsi,n_test-n_success+1)
            #optimal classifier
            opt_preds = opt_pred(dataset,X_test)
            n_success_opt += np.sum(opt_preds==y_test)
            clopper_pearson_interval[Ns+2] = beta.ppf(1-alpha/2,n_success_opt+1,n_test-n_success_opt+epsi) - beta.ppf(alpha/2,n_success_opt+epsi,n_test-n_success_opt+1)
            #clf combination with independent copula
            for i in range(y_test.size):
                combined_pred[i] = combinedPred(preds[:,i],cond_pdf,prior,copula='indep',theta=theta,cond_cdfs=cond_cdf)        
            n_success_ind += np.sum(combined_pred==y_test)
            clopper_pearson_interval[Ns+3] = beta.ppf(1-alpha/2,n_success_ind+1,n_test-n_success_ind+epsi) - beta.ppf(alpha/2,n_success_ind+epsi,n_test-n_success_ind+1)
            #clf combination based on weighted voting
            vote_preds = weighted_vote(preds,score_clf,n_class)
            n_success_vote += np.sum(vote_preds==y_test)
            clopper_pearson_interval[Ns+4] = beta.ppf(1-alpha/2,n_success_vote+1,n_test-n_success_vote+epsi) - beta.ppf(alpha/2,n_success_vote+epsi,n_test-n_success_vote+1)        
            #clf centralized learning
            ctl_preds = clf_ctl.predict(X_test)
            n_success_ctl += np.sum(ctl_preds==y_test)
            clopper_pearson_interval[Ns+5] = beta.ppf(1-alpha/2,n_success_ctl+1,n_test-n_success_ctl+epsi) - beta.ppf(alpha/2,n_success_ctl+epsi,n_test-n_success_ctl+1)
            #stacked clf 
            met_preds = clf_meta.predict(preds.T)
            n_success_met += np.sum(met_preds==y_test)
            clopper_pearson_interval[Ns+6] = beta.ppf(1-alpha/2,n_success_met+1,n_test-n_success_met+epsi) - beta.ppf(alpha/2,n_success_met+epsi,n_test-n_success_met+1)                
            for i in range(Ns+7):
                if (np.isnan(clopper_pearson_interval[i])):
                    clopper_pearson_interval[i]=1
            iter_nb += 1
            print("\r{}".format(iter_nb),'Largest Clopper-Pearson interval width:',np.max(clopper_pearson_interval),end="")
            sys.stdout.flush()
        print('\n')
        clf_rates /= n_test
        for i in range(Ns):
            print('Base classifier '+str(i)+' accuracy:',clf_rates[i])
        select_rate = n_success_select/n_test
        print('Selected classifier accuracy:',select_rate)
        vote_rate = n_success_vote/n_test
        print('Weighted vote accuracy:',vote_rate)
        ind_rate = n_success_ind/n_test
        print('DELCO (indep. cop.) accuracy:',ind_rate)
        combined_rate = n_success/n_test
        print('DELCO (Gaussian cop.) accuracy:',combined_rate)
        met_rate = n_success_met/n_test
        print('Stacking accuracy:',met_rate)
        ctl_rate = n_success_ctl/n_test
        print('Centralized classifier accuracy:',ctl_rate)
        opt_rate = n_success_opt/n_test
        print('Optimal classifier accuracy:',opt_rate)

        end = time.time()
        print('Elapsed time:',end-start)


    plt.figure()
    plt.scatter(X_all[:, 0], X_all[:, 1], marker='o', c=y_all, s=25,cmap='tab10')
    x_min = np.min(X_all[:, 0])
    if (x_min<0):
        x_min *=1.05
    else:
        x_min *=0.95     
    x_max = np.max(X_all[:, 0])
    if (x_max>0):
        x_max *=1.05
    else:
        x_max *=0.95     
    y_min = np.min(X_all[:, 1])
    if (y_min<0):
        y_min *=1.05
    else:
        y_min *=0.95     
    y_max = np.max(X_all[:, 1])
    if (y_max>0):
        y_max *=1.05
    else:
        y_max *=0.95     
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    if (dataset == 'blobs'):
        plt.plot([x_min,x_max],[y_max,y_min],'--m')
    elif (dataset == 'moons'):
        plt.plot([0,0],[y_max,y_min],'--m')
        plt.plot([1,1],[y_max,y_min],'--m')        
    elif (dataset == 'circles'):
        plt.plot([0,-y_min/np.sqrt(3)],[0,y_min],'--m')
        plt.plot([0,y_max/np.sqrt(3)],[0,y_max],'--m')
        plt.plot([0,x_min],[0,0],'--m')
    else:
        raise ValueError('Unknown generating process name.')    
        
    

    
if __name__ == "__main__": 
    
    ##############
    # PARAMETERS #
    ##############
    copula ='Gaussian' # 
    dataset = 'circles' # 'moons', 'circles', 'blobs'
    mode = 'deterministic_split' # 'random_split', 'deterministic_split'
    h = .02  # step size in the mesh
    n = 200 #  
    sent = 0.1
    iter_max = 20 # np.inf # 
    
    launch_test(dataset,mode,n,iter_max=iter_max,copula=copula)
    
    