import numpy as np
from .energy import energy_distance

def permutation_test_energy(X, Y, num_permutations=200, alpha=0.05):
    n = X.shape[0]
    m = Y.shape[0]
    Z = np.vstack([X, Y])
    observed_stat = energy_distance(X, Y)
    count = 0
    for _ in range(num_permutations):
        np.random.shuffle(Z)
        X_perm = Z[:n]
        Y_perm = Z[n:]
        stat = energy_distance(X_perm, Y_perm)
        if stat >= observed_stat:
            count += 1
    pval = (count + 1) / (num_permutations + 1)
    reject_null = pval < alpha
    return observed_stat, pval, reject_null, count


