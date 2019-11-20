
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def k_means_first():
    actual_data_points = np.asarray(
        [[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])
    plt.scatter(actual_data_points[:, 0], actual_data_points[:, 1], label='True Position')
    plt.show()

    # start points
    start_pts = np.array(
        [[2, 10], [5, 8], [1, 2]])

    for pair in actual_data_points:
        d1 =((pair[0] - start_pts[0][0]) ** 2 + (pair[1] - start_pts[0][1]) ** 2) ** 0.5
        d2 =((pair[0] - start_pts[1][0]) ** 2 + (pair[1] - start_pts[1][1]) ** 2) ** 0.5
        d3 =((pair[0] - start_pts[2][0]) ** 2 + (pair[1] - start_pts[2][1]) ** 2) ** 0.5
        print('{0:.2f}, {1:.2f}, {2:.2f}'.format(d1, d2, d3))


def k_means_final():
    # points in the question
    actual_data_points = np.asarray(
        [[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])
    plt.scatter(actual_data_points[:, 0], actual_data_points[:, 1], label='True Position')
    plt.show()

    # start points
    start_pts = np.array(
        [[2, 10], [5, 8], [1, 2]])

    # iterate five iterations
    number_iter= 5
    for i in range(number_iter):
        # define and train the model
        centroids = KMeans(n_clusters=3, init=start_pts, max_iter=1, n_init=1)
        centroids.fit(actual_data_points)

        print('------------------{}---------------'.format(i + 1))
        # get labels
        labels_ = centroids.labels_
        print(labels_)
        # plt.scatter(actual_data_points[:, 0], actual_data_points[:, 1], c=labels_, cmap='rainbow')
        # plt.show()

        # get the centroid array
        centroids_array = centroids.cluster_centers_
        print(centroids_array)

        # uodate the start_points
        start_pts = centroids_array


if __name__ == '__main__':
    k_means_first()
    k_means_final()


