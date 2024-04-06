import numpy as np

from mlib.base.base_regressor import BaseRegressor

class LinearRegression(BaseRegressor):
    def __init__(self, fit_intercept=True, normalize=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Input and output sizes don't match")

        if self.normalize:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        if self.fit_intercept:
            intercept = np.ones((X.shape[0],1))
            X = np.append(X, intercept, axis=1)

        XTX = np.dot(X.T, X)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(X.T, y)
        
        self.beta = np.dot(XTX_inv, XTy)

    def predict_reg(self, X):
        if self.normalize:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        if self.fit_intercept:
            intercept = np.ones((X.shape[0],1))
            X = np.append(X, intercept, axis=1)

        if X.shape[1] != self.beta.shape[0]:
            raise ValueError("Input matrix column size doesn't match coefficient vector size")

        return np.dot(X, self.beta)