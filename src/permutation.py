import numpy as np
from .mmd import compute_mmd_unbiased

def permutation_test(X, Y, kernel_func, B=300, alpha=0.05):
    n = X.shape[0]
    m = Y.shape[0]
    Z = np.vstack([X, Y])
    observed_mmd = compute_mmd_unbiased(X, Y, kernel_func)
    mmd_permutations = np.zeros(B)
    for b in range(B):
        np.random.shuffle(Z)
        X_perm = Z[:n]
        Y_perm = Z[n:n+m]
        mmd_permutations[b] = compute_mmd_unbiased(X_perm, Y_perm, kernel_func)
    p_value = np.mean(mmd_permutations >= observed_mmd)
    reject_null = p_value < alpha
    return observed_mmd, p_value, reject_null, mmd_permutations