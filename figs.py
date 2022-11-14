import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

atlas = "Dos160"
n_nodes = 160
results_clusters = f"./results/Node_{atlas}_C.node"
positions, clusters, size, labels = np.zeros((n_nodes,3)), np.zeros((n_nodes,)), np.zeros((n_nodes,)), []
with open(results_clusters) as f:
    for i, l in enumerate(f):
        line = l.split('\t')
        if i>0:
            positions[i-1,0], positions[i-1,1], positions[i-1,2] = float(line[0]), float(line[1]), float(line[2])
            clusters[i-1] = int(line[3])
            size[i-1] = int(line[3])
            labels.append(line[-1][:-1])
print(np.unique(clusters))
graph = np.loadtxt(f'./data/Edge_{atlas}_Binary.csv')
reordered_clusters = np.argsort(clusters)
reordered_labels = [labels[cl] for cl in reordered_clusters]
reordered = graph[reordered_clusters,:]
for i in range(reordered.shape[0]):
    reordered[i,:] = reordered[i,reordered_clusters]

initial = 0
for i, cl in enumerate(np.unique(clusters)):
    nodes_in_cluster = sum(clusters==cl)
    reordered[initial:initial+nodes_in_cluster,initial:initial+nodes_in_cluster] = reordered[initial:initial+nodes_in_cluster,initial:initial+nodes_in_cluster]*(2+i)
    initial += nodes_in_cluster

import matplotlib.colors
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["white","grey","red","gold","forestgreen","cornflowerblue","royalblue",
     "deeppink","darkorange","yellowgreen","seagreen"]
)

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(18,18))
plt.subplots_adjust(left=0.05,
                bottom=0.01, 
                right=0.99, 
                top=0.98)
ax.imshow(reordered, cmap=cmap)
ax.set_yticks(np.arange(0,n_nodes))
ax.set_yticklabels(reordered_labels,fontsize=8)
ax.set_xticklabels([])

plt.savefig(f"./BrainNet Figs/{atlas}-reordered.png", dpi=900)
plt.close()