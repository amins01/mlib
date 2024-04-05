import numpy as np

from mlib.base.base_classifier import BaseClassifier
from mlib.base.base_regressor import BaseRegressor

class KNN(BaseClassifier, BaseRegressor):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        if X.shape[0] < self.k or y.shape[0] < self.k:
            raise ValueError("Number of samples in the dataset must be >= k")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("Input and output sizes don't match")

        self.X_train = X
        self.y_train = y

    def predict_class(self, X):
        return self._predict(X, type="classification")

    def predict_reg(self, X):
        return self._predict(X, type="regression")

    def _predict(self, X, type):
        if X.shape[0] < self.k:
            raise ValueError("The number of samples in the dataset must be >= k")
        
        predications = []

        for i in range(X.shape[0]):
            sample = X[i]
            distances = {j: np.linalg.norm(sample - self.X_train[j]) for j in range(self.X_train.shape[0])}
            sorted_distances = {k: dist for k, dist in sorted(distances.items(), key=lambda item: item[1])}
            k_nearest_distances = dict(list(sorted_distances.items())[:self.k])
            k_nearest = [self.y_train[i] for i, _ in k_nearest_distances.items()]
            
            if type == "classification":
                unique_classes, counts = np.unique(k_nearest, return_counts=True)
                predications.append(unique_classes[np.argmax(counts)])
            elif type == "regression":
                predications.append(np.mean(k_nearest))

        return np.array(predications)