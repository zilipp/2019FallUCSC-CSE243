import math
import operator
import scipy.spatial.distance as dist
from sklearn import preprocessing
import numpy as np

X1 = [1.5, 1.7]
X2 = [2, 1.9]
X3 = [1.6, 1.8]
X4 = [1.2, 1.5]
X5 = [1.5, 1.0]
DATAS = [X1, X2, X3, X4, X5]
X_TEST = [1.4, 1.6]

#2.8(a)
#  Euclidean distance, Manhattan distance, supremum distance, and cosine similarity.
euclidean_map = dict()
manhattan_map = dict()
supremum_map = dict()
cosine_map = dict()


# euclidean
for i in range(len(DATAS)):
    dis = round(dist.euclidean(DATAS[i], X_TEST), 2)
    euclidean_map.update({'x' + str(i + 1): dis})

# manhattan
for i in range(len(DATAS)):
    dis = round(dist.cityblock(DATAS[i], X_TEST),2)
    manhattan_map.update({'x' + str(i + 1): dis})

# supremum
for i in range(len(DATAS)):
    dis = round(dist.chebyshev(DATAS[i], X_TEST), 2)
    supremum_map.update({'x' + str(i + 1): dis})

# cosine
for i in range(len(DATAS)):
    dis = round(1 - dist.cosine(DATAS[i], X_TEST), 4)
    cosine_map.update({'x' + str(i + 1): dis})

sorted_euclidean_map = sorted(euclidean_map.items(), key = operator.itemgetter(1))
sorted_manhattan_map = sorted(manhattan_map.items(), key = operator.itemgetter(1))
sorted_supremum_map = sorted(supremum_map.items(), key = operator.itemgetter(1))
sorted_cosine_map = sorted(cosine_map.items(), key = operator.itemgetter(1))
print(sorted_euclidean_map)
print(sorted_manhattan_map)
print(sorted_supremum_map)
print(sorted_cosine_map)


#2.8(b)
DATAS_norm = []
X_TEST_length = math.sqrt(X_TEST[0] ** 2 + X_TEST[1] ** 2)
X_TEST_norm = [round(X_TEST[0] / X_TEST_length, 4), round(X_TEST[1] / X_TEST_length, 4)]

for i in range(len(DATAS)):
    length = math.sqrt(DATAS[i][0] ** 2 + DATAS[i][1] ** 2)
    x_norm = [round(DATAS[i][0] / length, 4), round(DATAS[i][1] / length, 4)]
    DATAS_norm.append(x_norm)
print(DATAS_norm)

euclidean_norm_map = dict()
for i in range(len(DATAS_norm)):
    dis = round(dist.euclidean(DATAS_norm[i], X_TEST_norm), 4)
    euclidean_norm_map.update({'x' + str(i + 1): dis})
sorted_euclidean_norm_map = sorted(euclidean_norm_map.items(), key = operator.itemgetter(1))
print(sorted_euclidean_norm_map)