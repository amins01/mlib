import numpy as np

from mlib.base.base_clusterer import BaseClusterer
from mlib.utils import contains_point, euclidean_distance

class KMeans(BaseClusterer):
    def __init__(self, k, init="k-means++"):
        self.k = k
        self.init = init

    def fit(self, X, max_iter=500, tol=1e-4):
        if X.shape[0] < self.k:
            raise ValueError("Number of samples in the dataset must be >= k")
        
        self._init_centroids(X)
        self.inertia = float('inf')
        r = float('inf')

        for i in range(max_iter):
            clusters = [[] for _ in range(self.k)]
            
            for sample in X:
                closest_cluster = self._find_closest_cluster(sample)
                clusters[closest_cluster].append(sample)
            
            new_centroids = self._cluster_means(clusters, X)
            self.inertia = self._calculate_inertia(clusters)
            r = np.linalg.norm(new_centroids - self.centroids)
            print('inertia', self.inertia)

            if r < tol:
                print("Algorithm converged after {} iterations".format(i + 1))
                break

            self.centroids = np.array(new_centroids)

    def predict(self, X):
        return [self._find_closest_cluster(X[i]) for i in range(X.shape[0])]

    def _init_centroids(self, X):
        if self.init == "k-means++":
            self.centroids = [X[np.random.choice(X.shape[0])]]

            for _ in range(self.k - 1):
                distances = np.zeros((X.shape[0],))

                for c in self.centroids:
                    c_distances = [0 if contains_point(self.centroids, x) else euclidean_distance(c, x) for x in X]
                    farthest_point_index = np.argmax(c_distances)
                    distances[farthest_point_index] = max(distances[farthest_point_index], c_distances[farthest_point_index])

                self.centroids.append(X[np.argmax(distances)])

            self.centroids = np.array(self.centroids)
        else:
            # Randomly initialize cluster centroids
            self.centroids = X[np.random.choice(X.shape[0], size=self.k, replace=False)]

    def _calculate_inertia(self, clusters):
        inertia = 0

        for i, c in enumerate(clusters):
            for sample in c:
                inertia += np.linalg.norm(sample - self.centroids[i])**2

        return inertia

    def _find_closest_cluster(self, sample):
        sample_centroid_distances = [euclidean_distance(self.centroids[j], sample) for j in range(self.k)]
        return np.argmin(sample_centroid_distances)

    def _cluster_means(self, clusters, X):
        '''Needed due to the inhomogeneous shape of clusters'''
        means = []

        for c in clusters:
            if len(c) > 0:
                means.append(np.mean(c, axis=0))
            else:
                # Mean of an empty cluster = random point
                means.append(X[np.random.choice(X.shape[0])])

        return means