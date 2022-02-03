"""
Description:
By using the embeddings from the TAPNET framework a classification model is build 
and later evaluated

Input: csv file of embeddings
Output: induced model

@author: tlim3c
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path 
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
#from fastkde import fastKDE


print("?-read in data")
#Set this to the path you want to write your synthetic data
path = 'E:/SynologyDrive/programmering/eclipse-workspace/TapNet/plot/high/'
#name of data 
data = '_syn6_'
#part of path where to write the data
method = 'Syn6/'

embeddings_file = path + "embeddings.txt"
embeddings_df = pd.read_csv(Path(embeddings_file), header=None)

def interpolate_points(p1, p2, dist='uniform'):
    # interpolate points either using uniform distribution or
    if dist == 'uniform':
        rnd = np.random.uniform(low=0, high=1)
    else: # exponential distribution
        rnd = np.random.exponential(scale=0.1)
        #print('expontial gen')
    if rnd > 1: # sanity check
        rnd = 1
        #print('stopped at sanity-check')
    # linearly interpolate vectors
    v = (1.0 - rnd) * p1 + rnd * p2      
    return v

def syn_rnd_int(Xs, size=10):
    X = pd.DataFrame()
    no_ex = len(Xs) - 1
    for _ in range(0, size):
        ex = np.random.randint(low=1, high=no_ex) # note that 0 is reserved for the prototype
        syn_example = interpolate_points(Xs.iloc[0, :], Xs.iloc[ex, :])         # rnd uniform distribution
        #syn_example = interpolate_points(Xs.iloc[0, :], Xs.iloc[ex, :], 'exp') # rnd exponential distribution
        X = pd.concat([X, syn_example.to_frame().T], ignore_index=True)
    return X

def stepwise_syn_rnd_int(Xs, syn_size, step_size=1000): 
    Centroids = pd.DataFrame()
    if syn_size > step_size:
        curr_centroid = Xs.iloc[0, :]
        #print('First centroid: ', curr_centroid)
        X = pd.DataFrame()       
        no_ex = len(Xs) - 1
        for i in range(1, syn_size+1):
            (_,Dec) = divmod(i, step_size)
            if Dec == 0:            #update centroid
                batch_start = i-step_size
                batch_end = batch_start + step_size - 1
                X_batch = X.iloc[batch_start:batch_end, :]
                curr_centroid = X_batch.mean(axis=0) 
                Centroids = pd.concat([Centroids, curr_centroid.to_frame().T], ignore_index=True) 
                #print('New centroid: ', curr_centroid)
                ex = np.random.randint(low=1, high=no_ex) # note that 0 is reserved for the prototype
                syn_example = interpolate_points(curr_centroid, Xs.iloc[ex, :])       # rnd uniform distribution
                #syn_example = interpolate_points(Xs.iloc[0, :], Xs.iloc[ex, :], 'exp') # rnd exponential distribution
                X = pd.concat([X, syn_example.to_frame().T], ignore_index=True) 
            else:
                ex = np.random.randint(low=1, high=no_ex) # note that 0 is reserved for the prototype
                syn_example = interpolate_points(curr_centroid, Xs.iloc[ex, :])       # rnd uniform distribution
                #syn_example = interpolate_points(Xs.iloc[0, :], Xs.iloc[ex, :], 'exp') # rnd exponential distribution
                X = pd.concat([X, syn_example.to_frame().T], ignore_index=True) 
    else:
        X = syn_rnd_int(Xs, syn_size)
    return X, Centroids

def syn_kde(kde, size):
    X_syn = kde.sample(size, random_state=42)
    X = pd.DataFrame(X_syn)
    return X

def diff_check(X_syn, diff, pos_score, neg_score):
    del_index = []
    
    for idx, val in enumerate(pos_score):
        this_diff = val - neg_score[idx]
        if this_diff < diff:
            #remove these values
            del_index.append(idx)      
    return np.delete(X_syn, del_index, 0)       
    
def syn_kde_w_check(pos_kde, neg_kde, syn_size):
    X_syn = kde.sample(syn_size, random_state=42)
    pos_samp_score = pos_kde.score_samples(X_syn)
    neg_samp_score = neg_kde.score_samples(X_syn)
    
    pos_tot_score = pos_kde.score(X_syn) / syn_size
    neg_tot_score = neg_kde.score(X_syn) / syn_size
    
    diff = pos_tot_score - neg_tot_score
    
    keep_list = diff_check(X_syn, diff, pos_samp_score, neg_samp_score)
    
    while len(keep_list) < syn_size:
        new_syn = pos_kde.sample(1)
        pos_samp_score = pos_kde.score_samples(new_syn)[0]
        neg_samp_score = neg_kde.score_samples(new_syn)[0]  
        new_diff = pos_samp_score - neg_samp_score
        if new_diff > diff:
            keep_list = np.append(keep_list, new_syn, axis = 0)
    
    return keep_list 
     
def gen_smote(X_train, y_train, no_neg_ex, no_pos_ex, size):
    gen_size = no_pos_ex + size
    class_dict={0:no_neg_ex, 1:gen_size}
    X_smote, y_smote = SMOTE(sampling_strategy = class_dict).fit_resample(X_train, y_train)    
    X_syn = X_smote.tail(size)
    y_syn = y_smote.tail(size)
    return X_syn.to_numpy(), y_syn.to_list()

neg_ex = embeddings_df.loc[embeddings_df[0] == 0]
neg_X = neg_ex.drop(0, axis=1)       
pos_ex = embeddings_df.loc[embeddings_df[0] == 1]
pos_X = pos_ex.drop(0, axis=1)

#print('?-loding fitted kde models') 
#X_syn = pickle.load(open(path + 'X' + data + str(syn_size),'rb'))
#y_syn = pickle.load(open(path + 'y' + data + str(syn_size), 'rb'))
    
#print("?-fit standard PDF using KDE on positive data")
#?-best bandwidth: 0.09540954763499938, best leaf size: 15 

'''
params = {'bandwidth': np.logspace(-2, 1), 'leaf_size': np.arange(5, 100, 10)}
grid = GridSearchCV(KernelDensity(), params) # use gaussian kernel
grid.fit(pos_X) 
print("?-best bandwidth: {0}, best leaf size: {1}".format(grid.best_estimator_.bandwidth, grid.best_estimator_.leaf_size))
kde = grid.best_estimator_
'''
#De-comment this is you are using KDE or KDE_w_check
#kde = KernelDensity(bandwidth=0.09540954763499938, leaf_size=5).fit(pos_X)
#pickle.dump(kde, open(path + 'pos_kde', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)     
    
#print("?-fit standard PDF using KDE on negative data")
#?-best bandwidth: 0.3393221771895328, best leaf size: 15 

'''
params2 = {'bandwidth': np.logspace(-1, 1), 'leaf_size': np.arange(5, 100, 10)}
grid2 = GridSearchCV(KernelDensity(), params2) # use gaussian kernel
grid2.fit(neg_X) 
print("?-best bandwidth: {0}, best leaf size: {1}".format(grid2.best_estimator_.bandwidth, grid2.best_estimator_.leaf_size))
kde2 = grid2.best_estimator_    
'''
#De-comment this is you are using KDE_w_check
#kde2 = KernelDensity(bandwidth=0.3393221771895328, leaf_size=5).fit(neg_X)
#pickle.dump(kde2, open(path + 'neg_kde', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    
    
# Generate synthetic examples using: [ UI | EI | SUI | KDE | KDE_w_check | SMOTE ]
# Just de-comment out the method that you want to use! 
size_list = [10, 20, 40, 60, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 20000, 50000, 100000]
no_neg_ex = neg_X.shape[0]
no_pos_ex = pos_X.shape[0]
y_train = embeddings_df.loc[:,0]
X_train = embeddings_df.drop(0, axis=1)  
for syn_size in size_list:
    print('?-generate synthetic data of size: {0}'.format(syn_size))   
    #syn_X = syn_rnd_int(pos_X, syn_size)                                        # syn1 - UI & Syn2 - EI
    #syn_X, centroids = stepwise_syn_rnd_int(pos_X, syn_size, 1000)              # syn5 - SUI
    syn_X, pos_y = gen_smote(X_train, y_train, no_neg_ex, no_pos_ex, syn_size)   # syn6 - SMOTE  
    #syn_X = syn_kde(kde, syn_size)                                              # syn3 - KDE 
    #syn_X = syn_kde_w_check(kde, kde2, syn_size)                                # Syn4 - KDE with check
    #pos_y = [1]*syn_size

    #print("?-writing synthetic data to file")
    pickle.dump(syn_X, open(path + method + 'X'+ data + str(syn_size), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(pos_y, open(path + method + 'y' + data + str(syn_size), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    #pickle.dump(centroids, open(path + method + 'centroids'+ data + str(syn_size), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

print("?-finished")
