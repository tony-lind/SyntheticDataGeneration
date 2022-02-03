import pandas as pd
from pathlib import Path 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import pickle

def plot_acc(my_acc, size_list, folds):
    x = []
    y = []
    for size in size_list:
        x.append(size)
        acc = my_acc.get(size)/folds
        y.append(acc)
        print('Size: ', size)
        print('Balanced accuracy: ', acc)
    plt.plot(x, y)
    plt.ylim(bottom=0.5, top=0.85)
    plt.xlabel('Size')
    plt.ylabel('Balanced accuracy')
    plt.axhline(y=0.75, color='r', linestyle='-')
    plt.show()

print("?-read in data")
path = 'C:/Users/tony/eclipse-workspace-new/tapnet/plot/high/'

embeddings_file = path + "embeddings.txt"
embeddings_df = pd.read_csv(Path(embeddings_file), header=None)
#remove prototypes
X = embeddings_df.iloc[2:,:]

neg_ex = X.loc[X[0] == 0].to_numpy()       
pos_ex = X.loc[X[0] == 1].to_numpy()  #comment away if you use synthetic data

#de-comment for use of syntetic data
#pos_ex = pickle.load(open(path + 'X_syn6_20000','rb'))  
#y_syn = pickle.load(open(path + 'y_syn6_20000', 'rb'))

folds = 10
skf = StratifiedKFold(n_splits=folds)

size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 222]
my_acc={}
for size in size_list:   
    samp_pos = pos_ex[np.random.choice(pos_ex.shape[0], size, replace=False)]   #comment away if you use synthetic data
    samp_pos_y = samp_pos[:,0]                                                  #comment away if you use synthetic data
    samp_pos_X = np.delete(samp_pos, (0), axis=1)                               #comment away if you use synthetic data
    #de-comment for use with syntehtic data
    #samp_pos_X = pos_ex[np.random.choice(pos_ex.shape[0], size, replace=False)]
    #samp_pos_y = [1]*size   
    
    neg_y = neg_ex[:,0]
    neg_X = np.delete(neg_ex, (0), axis=1)
    X = np.concatenate([neg_X, samp_pos_X])
    y = np.concatenate([neg_y, samp_pos_y])
    print('Testing size of: ', str(size))
    for train, test in skf.split(X, y):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X[train],y[train])
        y_pred = clf.predict(X[test])
        score = balanced_accuracy_score(y[test], y_pred)
        print('Score: ', score)
        temp_acc = my_acc.get(size, 0)
        new_acc = temp_acc + score
        my_acc.update({size:new_acc})
        
plot_acc(my_acc, size_list, folds)        
        