import math
import numpy as np

from mlib.base.base_classifier import BaseClassifier
from mlib.helpers.data_preprocessing import DataPreprocessing

class NaiveBayes(BaseClassifier):
    
    def likelihood(self, sample, means, std_devs):
        '''Likelihood of a sample under a Gaussian distribution'''
        likelihoods = 1 / np.sqrt(2 * math.pi * np.square(std_devs)) * np.exp(-np.square(sample - means) / (2 * np.square(std_devs)))
        return np.prod(likelihoods)

    def fit(self, X, y):
        self.means_by_class = {}
        self.std_devs_by_class = {}
        self.X_train = X
        self.y_train = y
        self.classes, classes_counts = np.unique(y, return_counts=True)
        self.priors = classes_counts / len(y)

        Xy_train = np.column_stack((X, y))

        for c in self.classes:
            # Create a mask for the samples of that class
            c_mask = Xy_train[:, -1] == c
            means = np.mean(Xy_train[c_mask], axis=0)[:-1]
            std_devs = np.std(Xy_train[c_mask], axis=0)[:-1]
            self.means_by_class[c] = means
            self.std_devs_by_class[c] = std_devs

    def predict_class(self, X):
        pred = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            sample = X[i]
            posteriors = np.zeros_like(self.classes)
            
            for c_i, c in enumerate(self.classes):
                likelihood = self.likelihood(sample, self.means_by_class[c], self.std_devs_by_class[c])
                posterior = likelihood * self.priors[c_i]
                posteriors[c_i] = posterior

            pred[i] = self.classes[np.argmax(posteriors)]

        return pred