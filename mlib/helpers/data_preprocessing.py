import numpy as np

class DataPreprocessing():

    @staticmethod
    def vector_to_one_hot_matrix(v, num_cols):
        one_hot_m = np.zeros((v.shape[0], num_cols))
        one_hot_m[np.arange(v.shape[0]), v] = 1
        return one_hot_m
