class BaseClusterer:
    def __init__(self):
        pass

    def fit(self, X):
        raise NotImplementedError("Method not implemented")

    def predict(self, X):
        raise NotImplementedError("Method not implemented")