import numpy as np

def vector_to_one_hot_matrix(v, num_cols):
        one_hot_m = np.zeros((v.shape[0], num_cols))
        one_hot_m[np.arange(v.shape[0]), v] = 1
        return one_hot_m

def shuffle(a, b):
    if(a.shape[0] != b.shape[0]):
        raise ValueError("Arrays don't have the same number of rows")
    
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)

    return a[indices], b[indices]

def logsumexp(a, axis, keepdims, stable):
    '''Numerically stable log of sum of exponentials'''
    max = np.max(a, axis=axis, keepdims=keepdims) if stable else np.zeros_like(a)
    sum = np.sum(np.exp(a - max), axis=axis, keepdims=keepdims)

    return np.log(sum) + max

def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def contains_point(X, p):
    for x in X:
        if all(x[i] == p[i] for i in range(len(x))):
            return True
    return False