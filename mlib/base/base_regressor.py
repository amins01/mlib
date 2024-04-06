class BaseRegressor:
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError("Method not implemented")

    def predict_reg(self, X):
        raise NotImplementedError("Method not implemented")