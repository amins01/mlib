import numpy as np

from mlib.base.base_clusterer import BaseClusterer
from mlib.utils import euclidean_distance

class KMeans(BaseClusterer):
    def __init__(self, k):
        self.k = k

    def fit(self, X, max_iter=500, tol=1e-4):
        if X.shape[0] < self.k:
            raise ValueError("Number of samples in the dataset must be >= k")
        
        # Randomly initialize cluster centroids
        self.centroids = X[np.random.choice(X.shape[0], size=self.k, replace=False)]
        r = float('inf')

        for i in range(max_iter):
            clusters = [[] for _ in range(self.k)]
            
            for sample in X:
                closest_cluster = self._find_closest_cluster(sample)
                clusters[closest_cluster].append(sample)
            
            new_centroids = self._cluster_means(clusters, X)
            r = np.linalg.norm(new_centroids - self.centroids)
            print('r', r)

            if r < tol:
                print("Algorithm converged after {} iterations".format(i + 1))
                break

            self.centroids = np.array(new_centroids)

    def predict(self, X):
        return [self._find_closest_cluster(X[i]) for i in range(X.shape[0])]

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