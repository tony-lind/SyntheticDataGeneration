import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from pathlib import Path 
#from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
#
Seed = 42

# Utility function to visualize the outputs of PCA and t-SNE
def tsne_scatter_plot(syn_size, x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.set_title('Syntetic size: ' + str(syn_size))
    #sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    sc = ax.scatter(x[2:,0], x[2:,1], lw=0, s=40, c=palette[colors[2:].astype(np.int)])
    
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    
    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    
    # ignore this?    
    for i in range(1, syn_size+1):
        step_size=1000
        (_,Dec) = divmod(i, step_size)
        if Dec == 0 and i != 1000:            #update centroid
            batch_start = i-step_size
            batch_end = batch_start + step_size - 1
            X_batch = x[batch_start:batch_end, :]
            x_cent, y_cent = X_batch.mean(axis=0)
            txts.append(ax.text(x_cent, y_cent, 'c'+str(int(i/1000)), fontsize=24))
        
        txts.append(ax.text(x[0,0], x[0,1], '*', fontsize=24))
        txts.append(ax.text(x[1,0], x[1,1], '^', fontsize=24))
    return f, ax, sc, txts

print("?-read in data")
#Set this to the path where you have your synthetic data
path = 'E:/SynologyDrive/programmering/eclipse-workspace/TapNet/plot/high/'

embeddings_file = path + "embeddings.txt"
embeddings_df = pd.read_csv(Path(embeddings_file), header=None)
y_org = embeddings_df.iloc[:,0].to_numpy()
X_org = embeddings_df.drop(0,axis=1).to_numpy()
y = y_org #y_org[2:]           #remove first two examples (prototypical examples)
X = X_org #X_org[2:, :]

print('No classes: ', np.unique(y))

size_list = [100, 200, 400, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 20000, 50000, 100000]
data = '_syn5_'             #name of data 
method = 'Syn5/'            #part of path where to write the data

for syn_size in size_list:
    X_concat = None
    y_concat = None
    print('?-merging original data with synthetic data') 
    X_syn = pickle.load(open(path + method + 'X' + data + str(syn_size),'rb'))
    #y_syn = pickle.load(open(path + method + 'y' + data + str(syn_size), 'rb'))
    y_syn = [2]*syn_size
    
    X_concat = np.concatenate([X, X_syn])
    y_concat = np.concatenate([y, y_syn])

    # run t-SNE
    time_start = time.time()
    tsne = TSNE(random_state=Seed).fit_transform(X_concat)
    tsne
    print('?-t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    # plot t-SNE results (X in two dimensions and y color coded)
    f_tsne, ax_tsne, sc_tsne, txts_tsne = tsne_scatter_plot(syn_size, tsne, y_concat) # Visualizing the PCA output
    print('?-t-SNE plot')
    #f_tsne.show()
    f_tsne.savefig(path + method + 't_SNE' + '_3class_' + str(syn_size))
    #plt.savefig(path + method + 't_SNE' + str(syn_size))
print('?-plotting done')