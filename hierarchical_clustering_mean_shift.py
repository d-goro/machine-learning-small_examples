#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 18:49:56 2018

@author: dimm
"""

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

centers = [[1, 1], [5, 5], [3, 10]]

#underscore _ - says it's useless, pay no attention
X, _ =  make_blobs(n_samples = 500, centers = centers, cluster_std = 1)

plt.scatter(X[:,0], X[:,1])
plt.show()

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)

n_clusters_ = len(np.unique(labels)) #how many categories (cluster types) machine returned to us

print('Number of estimated clusters:', n_clusters_)

colors = 10 * ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']

print(colors)
print(labels)

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
plt.show()   