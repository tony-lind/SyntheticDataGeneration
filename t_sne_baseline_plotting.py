import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from pathlib import Path 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
#
Seed = 42

# Utility function to visualize the outputs of PCA and t-SNE
def tsne_scatter_plot(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    #sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    ax.scatter(x[0,0], x[0,1], lw=0, s=40, marker='D')
    ax.scatter(x[1,0], x[1,1], lw=0, s=40, marker='D')
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

        txts.append(ax.text(x[0,0], x[0,1], '*', fontsize=24))
        txts.append(ax.text(x[1,0], x[1,1], '^', fontsize=24))
    return f, ax, sc, txts

print("?-read in data")
path = 'E:/SynologyDrive/programmering/eclipse-workspace/TapNet/plot/low/'

embeddings_file = path + "embeddings.txt"
embeddings_df = pd.read_csv(Path(embeddings_file), header=None)
y_org = embeddings_df.iloc[:,0].to_numpy()
X_org = embeddings_df.drop(0,axis=1).to_numpy()
y = y_org #y_org[2:]           #remove first two examples (prototypical examples)
X = X_org #X_org[2:, :]

print(np.unique(y))

# run PCA 
time_start = time.time()
pca = PCA(n_components=4)
pca_result = pca.fit_transform(X)
print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))

# print PCA results
pca_df = pd.DataFrame()
pca_df['pca1'] = pca_result[:,0]
pca_df['pca2'] = pca_result[:,1]
pca_df['pca3'] = pca_result[:,2]
pca_df['pca4'] = pca_result[:,3]
print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))

# plot PCA results (X in two dimensions and y color coded)
#top_two_comp = pca_df[['pca1','pca2']] # taking first and second principal component
#f_pca, ax_pca, sc_pca, txts_pca = fashion_scatter(top_two_comp.values, y) # Visualizing the PCA output
#print('PCA plot')
#f_pca.show()

# run t-SNE
time_start = time.time()
tsne = TSNE(random_state=Seed).fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# plot t-SNE results (X in two dimensions and y color coded)
f_tsne, ax_tsne, sc_tsne, txts_tsne = tsne_scatter_plot(tsne, y) # Visualizing the PCA output
print('t-SNE plot')
#f_tsne.show()
f_tsne.savefig(path +'t_SNE_no_syn_data')
print('stop')