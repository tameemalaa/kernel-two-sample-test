import numpy as np

def energy_distance(X, Y):
    n, m = len(X), len(Y)
    
    def avg_dist(A, B):
        dists = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        return np.sum(dists) / (len(A)*len(B))
    
    term1 = avg_dist(X, Y)
    term2 = avg_dist(X, X)
    term3 = avg_dist(Y, Y)
    energy = 2 * term1 - term2 - term3
    return energy