import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


prices = [5, 10, 11, 13, 15, 35, 50, 55, 72, 92, 204, 215]
# 3.9(a)
print('bin1: 5, 10, 11, 13')
print('bin2: 15, 35, 50, 55')
print('bin3: 72, 92, 204, 215')

# 3.9(b)
print('bin1[5, 75): 5, 10, 11, 13')
print('bin2[75, 145): 15, 35, 50, 55')
print('bin3[145-215]: 72, 92, 204, 215')

# 3.9(c)
kmeans = KMeans(n_clusters=2)
kmeans.fit(np.asarray(prices).reshape(-1, 1))
print(kmeans.cluster_centers_)
print(kmeans.labels_)


