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
from pathlib import Path 
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from math import sqrt


def find_threshold(model, X_val, y_val, cost_dict, plot=False, step_size=1, verbose=True):
    best_threshold = None
    best_cost = None
    best_dict = None
    steps = 100 // step_size
    problist = model.predict_proba(X_val)    
    plot_df = pd.DataFrame(columns=['threshold', 'cost', 'fn', 'fp'])
    for threshold in range(0, steps):      
        y_pred = apply_threshold(threshold, problist)
        cm = confusion_matrix(y_val, y_pred)
        tp = cm[0, 0]
        fp = cm[0, 1] 
        fn = cm[1, 0]
        tn = cm[1, 1]        
        cost = cost_dict.get('false_0')*fn + cost_dict.get('false_1')*fp
        if(verbose):
            print('Cost for threshold {} is cost {}'.format(threshold, cost))
            print('False positive {} and false negative {}'.format(fp, fn))
        if(plot):
            new_row = {'threshold': threshold, 'cost':cost, 'fp': fp, 'fn': fn}
            plot_df = plot_df.append(new_row, ignore_index=True)
        if best_cost == None:
            best_cost = cost
            best_threshold = threshold
        elif best_cost > cost:
            best_cost = cost
            best_dict = new_row
            best_threshold = threshold        
    if(plot):
        print('do plotting 1')
        x_plot = plot_df['threshold'].values.tolist()
        y_plot = plot_df['cost'].values.tolist()
        plt.plot(x_plot, y_plot)
        plt.title('cost-decision visualization')
        plt.xlabel('Decision threshold')
        plt.ylabel('Cost')
        ymin = plot_df['cost'].min()
        ymax = plot_df['cost'].max()
        plt.vlines(x= best_threshold, ymin=ymin, ymax=ymax, colors="red", label="best threshold")
        plt.show()
        
        print('do plotting 2')
        x_plot = plot_df['threshold'].values.tolist()
        y_plot_fp = plot_df['fp'].values.tolist()
        y_plot_fn = plot_df['fn'].values.tolist()
        _, ax = plt.subplots()
        ax.plot(x_plot, y_plot_fp, color='red')
        ax.set_ylabel('false positive', color='red')
        ax.set_xlabel('Decision threshold')
        #plot false negatives
        ax2 = ax.twinx()
        ax2.plot(x_plot, y_plot_fn, color='blue')
        ax2.set_ylabel('false negative', color='blue')
        plt.show()
    return best_dict

def find_threshold_problist(X_val, y_val, cost_dict, step_size=1, verbose=True):
    best_threshold = None
    best_cost = None
    best_cm = None
    best_pred = None 
    steps = 100 #// step_size 
    for threshold in range(0, steps):      
        y_pred = apply_threshold(threshold, X_val)
        cm = confusion_matrix(y_val, y_pred)
        tp = cm[0, 0]
        fn = cm[0, 1] 
        fp = cm[1, 0]
        tn = cm[1, 1]        
        cost = cost_dict.get('false_0')*fn + cost_dict.get('false_1')*fp
        if(verbose):
            print('Cost for threshold {} is cost {}'.format(threshold, cost))
            print('False positive {} and false negative {}'.format(fp, fn))
        if best_cost == None:
            best_cost = cost
            best_cm = cm
            best_threshold = threshold
            best_pred = y_pred 
        elif best_cost > cost:
            best_cost = cost
            best_cm = cm
            best_threshold = threshold 
            best_pred = y_pred       
    return best_cm, best_threshold, best_cost, best_pred

def apply_threshold(threshold, problist):
    prediction_list = []
    #using threshold for pos class = 1    
    for row in problist:
        if row[0] >= threshold/100:
            prediction_list.append(0)
        else: 
            prediction_list.append(1)        
    return prediction_list 


def test_data(clf, X, y, syn_X=None, syn_y=None):
    k_folds = 10
    skf = StratifiedKFold(k_folds)
    threshold, cost, acc, bacc, con_mat, auc = [], [], [], [], [], []
    for train, test in skf.split(X, y):
        xt, xv, yt, yv = X[train, :], X[test, :], y[train], y[test]
        if syn_y != None and len(syn_y) > 0:           
            xt = np.concatenate([xt, syn_X])
            yt = np.concatenate([yt, syn_y])
            print('size of concatenated data is: {0}'.format(len(yt)))
        clf.fit(xt, yt)
        proba = clf.predict_proba(xv)
        #yhat = clf.predict(xv)
        best_cm, best_threshold, best_cost, best_pred = find_threshold_problist(proba, yv, {'false_0':500, 'false_1':10}, verbose=False) 
        threshold.append(best_threshold)
        cost.append(best_cost)
        acc.append(accuracy_score(yv, best_pred))      
        bacc.append(balanced_accuracy_score(yv, best_pred))
        #con_mat.append(confusion_matrix(yv, yhat))
        con_mat.append(best_cm)
        auc.append(roc_auc_score(yv, proba[:, 1]))

    threshold_mean, threshold_std = np.mean(threshold), np.std(threshold)
    cost_mean, cost_std = np.mean(cost), np.std(cost)
    acc_mean, acc_std = np.mean(acc), np.std(acc)
    bacc_mean, bacc_std = np.mean(bacc), np.std(bacc)
    auc_mean, auc_std = np.mean(auc), np.std(auc)
    
    tp, fp, fn, tn = 0, 0, 0, 0
    for con in con_mat:
        tp = tp + con[0][0]
        fn = fn + con[0][1]
        fp = fp + con[1][0]        
        tn = tn + con[1][1]
    tp_mean = tp / k_folds
    fn_mean = fn / k_folds 
    fp_mean = fp / k_folds    
    tn_mean = tn / k_folds    
        
    tp_dev, tn_dev, fp_dev, fn_dev = 0, 0, 0, 0
    for con in con_mat:
        tp_dev = tp_dev + pow(tp_mean - con[0][0], 2)       
        fn_dev = fn_dev + pow(fn_mean - con[0][1], 2) 
        fp_dev = fp_dev + pow(fp_mean - con[1][0], 2)
        tn_dev = tn_dev + pow(tn_mean - con[1][1], 2)
    tp_std = sqrt(tp_dev / k_folds)
    fn_std = sqrt(fn_dev / k_folds)
    fp_std = sqrt(fp_dev / k_folds)  
    tn_std = sqrt(tn_dev / k_folds)     
    
    return {'threshold_mean': threshold_mean,
            'threshold_std': threshold_std,
            'cost_mean': cost_mean,
            'cost_std': cost_std,
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'bacc_mean': bacc_mean,
            'bacc_std': bacc_std,
            'auc_mean': auc_mean,
            'auc_std': auc_std, 
            'tp_mean': tp_mean,
            'tp_std': tp_std,
            'fp_mean': fp_mean,
            'fp_std': fp_std,
            'fn_mean': fn_mean,
            'fn_std': fn_std,
            'tn_mean': tn_mean,
            'tn_std': tn_std}

#Set this to the path you want to write your synthetic data
print("?-read in data")
path = 'E:/SynologyDrive/programmering/eclipse-workspace/TapNet/plot/high/'
data = '_syn5_'
method = 'Syn5/'

size_list = [100, 200, 400, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 20000, 50000, 100000]
#size_list = [1000]
f = open('result' + data, 'a')

# with embeddings

embeddings_file = path + "embeddings.txt"
embeddings_df = pd.read_csv(Path(embeddings_file), header=None)
y = embeddings_df.iloc[:,0].to_numpy()
X = embeddings_df.drop(0,axis=1).to_numpy()
y = y[2:]           #remove first two examples (prototypical examples)
X = X[2:, :]

for syn_size in size_list:
    print('?-training RF model using synthetic data of size: {0}'.format(syn_size)) 
    X_syn = pickle.load(open(path + method + 'X' + data + str(syn_size),'rb'))  #X_syn
    y_syn = pickle.load(open(path + method + 'y' + data + str(syn_size), 'rb')) #y_syn
    
    print("?-train model on our data")
    rf_model = RandomForestClassifier(n_jobs=-1)
    
    print("?-do experiment")
    p_dict =  test_data(rf_model, X, y, X_syn, y_syn)
    
    print("?-writing results to file")
    f.write('synthetic size: {0}\n'.format(syn_size))
    f.write('threshold: {0:.3f} +/- {1:.3f}, '.format(p_dict.get('threshold_mean'),p_dict.get('threshold_std')))
    f.write('cost: {0:.3f} +/- {1:.3f}, '.format(p_dict.get('cost_mean'),p_dict.get('cost_std')))
    f.write('accuracy: {0:.3f} +/- {1:.3f}, '.format(p_dict.get('acc_mean'), p_dict.get('acc_std')))
    f.write('balanced accuracy: {0:.3f} +/- {1:.3f}, '.format(p_dict.get('bacc_mean'), p_dict.get('bacc_std')))
    f.write('auc: {0:.3f} +/- {1:.3f}\n'.format(p_dict.get('auc_mean'), p_dict.get('auc_std')))
    f.write('tp: {0:.3f} +/- {1:.3f} |'.format(p_dict.get('tp_mean'), p_dict.get('tp_std')))
    f.write('fn: {0:.3f} +/- {1:.3f}\n'.format(p_dict.get('fn_mean'), p_dict.get('fn_std')))
    f.write('fp: {0:.3f} +/- {1:.3f} |'.format(p_dict.get('fp_mean'), p_dict.get('fp_std')))
    f.write('tn: {0:.3f} +/- {1:.3f}\n'.format(p_dict.get('tn_mean'), p_dict.get('tn_std')))   
    
    pickle.dump(p_dict, open('result' + data + str(syn_size), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
          
f.close()    

print("?-finished")

