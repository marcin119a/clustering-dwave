from email.base64mime import header_length
from scipy.sparse.csgraph import laplacian
import numpy as np
import pandas as pd
import networkx as nx

atlas = 'karate'
#df = np.genfromtxt(f'data/Edge_{atlas}_Binary.csv', delimiter='	')
G = nx.karate_club_graph()
df = nx.to_numpy_matrix(G)
print(df.shape)
L = laplacian(df, normed=True)

eigenvalues, v = np.linalg.eig(L)
eigenvalues = np.sort(eigenvalues)

max_gap = 0
gap_pre_index = 0
gap=[0] * eigenvalues.size
for i in range(1, eigenvalues.size):
        gap[i-1] = eigenvalues[i] - eigenvalues[i - 1]
        if gap[i-1] > max_gap and i<20:
            max_gap = gap[i-1]
            gap_pre_index = i - 1

k = gap_pre_index + 1
k2, max_gap_2 = np.argsort(gap)[-2]+1, np.sort(gap)[-2]
k3, max_gap_3 = np.argsort(gap)[-3]+1, np.sort(gap)[-3]
print("First GAP at k = ", k, "GAP = ", max_gap)
print("Second GAP at k = ", k2, "GAP_2 = ", max_gap_2)
print("Third big GAP at k = ", k3, "GAP_3 = ", max_gap_3)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4))
plt.subplots_adjust(left=0.12,
                bottom=0.12, 
                right=0.98, 
                top=0.98)
plt.gcf().text(0.32, 0.915, r"$\tilde{k}_1$" + f"$={str(k)}$", fontsize=15, fontweight="bold")
plt.gcf().text(0.35, 0.8, r"$\tilde{k}_2$" + f"$={str(k2)}$", fontsize=15, fontweight="bold")
ax.arrow(k+.8,max_gap,2,0, head_width = 0.005, head_length=2, color='red')
ax.arrow(k2+.8,max_gap_2,2,0, head_width = 0.005, head_length=2, color='blue')

ax.plot(range(1,eigenvalues.size+1), gap, marker='o', linestyle='solid', linewidth=1, markersize=4, color='black') 
ax.vlines(k, ymin=0, ymax=max_gap-0.012, linewidth=0.75, linestyle='dashed', color='red')
ax.vlines(k2, ymin=0, ymax=max_gap_2-0.012, linewidth=0.75, linestyle='dashed', color='blue')

ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
ax.set_ylabel("$\lambda_{k+1} - \lambda_k$", fontsize=14), ax.set_xlabel("k", fontsize=14)
ax.set_xticks([1,20,40,60,80,100,120,140,160]), ax.set_xlim([0.5,eigenvalues.size+1]), ax.set_ylim([0,np.max(gap)+0.01])
ax.set_xticklabels(["1","20","40","60","80","100","120","140","160"])

left, bottom, width, height = [0.58, 0.6, 0.36, 0.36]
inset = fig.add_axes([left, bottom, width, height])
inset.scatter(range(1,eigenvalues.size+1), eigenvalues, s=2, color='black')
inset.scatter(k, eigenvalues[k-1], s=4, color='red')
inset.scatter(k2, eigenvalues[k2-1], s=4, color='blue')
inset.vlines(k, ymin=0, ymax=eigenvalues[k-1]-0.012, linewidth=0.5, linestyle='dashed', color='red')
inset.vlines(k2, ymin=0, ymax=eigenvalues[k2-1]-0.012, linewidth=0.5, linestyle='dashed', color='blue')

inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
inset.set_ylabel("$\lambda_k$", fontsize=10), inset.set_xlabel("k", fontsize=10)
inset.set_xticks([1,20,40,60,80,100,120,140,160]), inset.set_xlim([0,eigenvalues.size+1]), inset.set_ylim([0,np.max(eigenvalues)+0.1])


plt.savefig(f"./results/Eigengap_{atlas}.png", dpi=900)