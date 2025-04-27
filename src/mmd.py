import numpy as np

def compute_mmd_unbiased(X, Y, kernel_func):
    n = X.shape[0]
    m = Y.shape[0]
    K_XX = kernel_func(X, X)
    K_YY = kernel_func(Y, Y)
    K_XY = kernel_func(X, Y)
    sum_K_XX = (np.sum(K_XX) - np.sum(np.diag(K_XX))) / (n * (n - 1))
    sum_K_YY = (np.sum(K_YY) - np.sum(np.diag(K_YY))) / (m * (m - 1))
    sum_K_XY = np.sum(K_XY) / (n * m)
    return sum_K_XX + sum_K_YY - 2 * sum_K_XY