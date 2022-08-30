from clustering import modularization, COLORS 
import networkx as nx
import numpy as np 
from numpy import genfromtxt
import csv
import time
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm

def main(folder):
    A = genfromtxt('data/Edge_AAL90_Binary.csv', delimiter='	')
    G = nx.from_numpy_matrix(A)
    B = nx.modularity_matrix(G)
    

    for k in [4, 11]:
        for r in range(10):
            communities, run_time, energy, counts, sample = modularization(G, B, k) #a former version has been used with additional parameters B,k; basic algorithm the same as in later evaluation
            communities_class = nx_comm.louvain_communities(G, seed=123)

            start = time.time()
            nx_comm.modularity(G, nx_comm.label_propagation_communities(G))
            end = time.time()
            total_time = end - start

            data = [communities, run_time, energy, counts, sample, communities_class, total_time]
            with open(f'{folder}/run{k}_{r}.csv', 'w') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            file.close()

            color_map = []

            for node in G:
                color_map.append(COLORS[sample[node]])
            f = plt.figure()

            nx.draw(G, node_color=color_map, with_labels=True, ax=f.add_subplot(111))
            f.savefig(f"{folder}/graph{k}_{r}.png")

            clus = np.zeros((len(G.nodes), 2))

            for i, node in enumerate(G):
                clus[i, 0] = node
                clus[i, 1] = sample[node]

            np.savetxt(f"{folder}/clustering{k}_{r}.csv", clus, delimiter=",")

if __name__ == '__main__':
    main(folder='output')