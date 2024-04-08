import numpy as np

from mlib.base.base_classifier import BaseClassifier
from mlib.helpers.data_preprocessing import DataPreprocessing

class LogisticRegression(BaseClassifier):
    def __init__(self, learning_rate = 0.05):
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e_z / np.sum(e_z, axis=-1, keepdims=True)

    def cross_entropy_loss(self, X, y, y_pred):
        m = X.shape[0]
        return (-1/m) * np.sum(y * np.log(y_pred))

    def fit(self, X, y, max_iter=1000, tol=1e-4):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Input and output sizes don't match")

        m = X.shape[0]
        num_classes = len(np.unique(y))
        y = DataPreprocessing.vector_to_one_hot_matrix(y, num_classes)
        self.weights = np.zeros((X.shape[1], num_classes))
        self.biases = np.zeros(num_classes)

        prev_loss = np.inf
        for i in range(max_iter):
            h = self._predict(X)

            loss = self.cross_entropy_loss(X, y, h)
            print("Loss at epoch {}: {}".format(i + 1, loss))

            if abs(prev_loss - loss) < tol:
                print("Algorithm converged after {} iterations".format(i + 1))
                break

            self.weights -= self.learning_rate * 1/m * np.dot(X.T, (h - y))
            self.biases -= self.learning_rate * np.mean(h - y, axis=0)

            prev_loss = loss

    def predict_class(self, X, proba=False):
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError("Input matrix column size doesn't match coefficient vector size")

        if proba:
            return self._predict(X)

        return np.argmax(self._predict(X), axis=1)
    
    def _predict(self, X):
        return self.softmax(np.dot(X, self.weights + self.biases))