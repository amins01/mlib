import math
import numpy as np

from mlib.base.base_classifier import BaseClassifier

class NaiveBayes(BaseClassifier):
    
    def fit(self, X, y):
        self.means_by_class = {}
        self.std_devs_by_class = {}
        self.classes, classes_counts = np.unique(y, return_counts=True)
        self.priors = classes_counts / len(y)

        # We combine X and y to allow us to filter samples by output class
        Xy_train = np.column_stack((X, y))

        for c in self.classes:
            class_mask = Xy_train[:, -1] == c # Create a boolean mask for the samples of that class
            class_samples = Xy_train[class_mask]
            means = np.mean(class_samples, axis=0)[:-1]
            std_devs = np.std(class_samples, axis=0)[:-1]
            self.means_by_class[c] = means
            self.std_devs_by_class[c] = std_devs

    def predict_class(self, X):
        pred = np.zeros(X.shape[0])

        for i, sample in enumerate(X):
            posteriors = np.zeros_like(self.classes)
            
            for c_i, c in enumerate(self.classes):
                log_likelihood = self._log_likelihood(sample, self.means_by_class[c], self.std_devs_by_class[c])
                posterior = log_likelihood + np.log(self.priors[c_i]) # log(likelihood * prior) = log(likelihood) + log(prior)
                posteriors[c_i] = posterior

            pred[i] = self.classes[np.argmax(posteriors)]

        return pred
    
    def _log_likelihood(self, sample, means, std_devs):
        '''Likelihood of a sample under a Gaussian distribution.
            We're working in log space to prevent underflow'''
        log_likelihoods = -0.5 * np.log(2 * math.pi * np.square(std_devs)) - np.square(sample - means) / (2 * np.square(std_devs))
        return np.sum(log_likelihoods) # log(m * n) = log(m) + log(n)