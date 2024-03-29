# -*- coding: utf-8 -*-
"""FinalClustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15bMAU3FrXFr_VCSgXfo5U7ao_RJnJq1o
"""


import networkx as nx
import numpy as np
from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler

COLORS = {0: 'blue', 1: 'red', 2: '#2a401f', 3: '#cce6ff',
          4:'pink', 5: '#4ebd1a', 6: '#66ff66', 7:'yellow',
          8: '#0059b3', 9: '#703243', 10: 'green', 11: 'black',
          12:'#3495eb', 13: '#525c4d', 14: '#1aff1a', 15: 'brown', 16: 'gray'}


def modularization(G, B, num_partitions):
  partitions = range(num_partitions)
  # Initialize the DQM object
  dqm = DiscreteQuadraticModel()

  for i in G.nodes():
      dqm.add_variable(num_partitions, label=i)

  for i in G.nodes():
      for j in G.nodes():
          if i==j:
              continue #the algorithm skips the linear term in QUBO/Ising formulation as in k-community a node has to belong to one community, therefore there is no effect in the maximising constellation
          dqm.set_quadratic(i,j, {(c, c): ((-1)*B[i,j]) for c in partitions})


  # Initialize the DQM solver
  sampler = LeapHybridDQMSampler(token='here token for D-WAVE')

  # Solve the problem using the DQM solver
  sampleset = sampler.sample_dqm(dqm)

  # get the first solution
  sample = sampleset.first.sample
  energy = sampleset.first.energy

  run_time=(sampleset.info['run_time'])*0.001 #total runtime in milliseconds

  # Count the nodes in each partition
  counts = np.zeros(num_partitions)
      
  #create communities as parameter for evaluation function
  communities=[]
  for k in partitions:
      comm=[]
      for i in sample:
          if sample[i]==k:
              comm.append(i)
      communities.append(set(comm))

  #compute number of nodes in each community
  for i in sample:
      counts[sample[i]] += 1
  
  return (communities, run_time, energy, counts,sample)



