import numpy as np
import matplotlib.pyplot as plt
from src.kernel import gaussian_rbf_kernel
from src.permutation import permutation_test

# Parameters
np.random.seed(42)
dims = [2, 5, 10, 20]  # Different dimensionalities
bandwidths = np.linspace(0.1, 5, 10)  # Different kernel bandwidths
n_samples = 50
delta = 0.5
n_trials = 20
B = 300
alpha = 0.05

# Results storage
results_h0 = np.zeros((len(dims), len(bandwidths)))
results_h1 = np.zeros((len(dims), len(bandwidths)))

# Ablation over dimensionality and kernel bandwidth
for i, dim in enumerate(dims):
    for j, sigma in enumerate(bandwidths):
        # Synthetic H0: Identical distributions
        rejections_h0 = 0
        for _ in range(n_trials):
            X = np.random.normal(0, 1, size=(n_samples, dim))
            Y = np.random.normal(0, 1, size=(n_samples, dim))
            kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=sigma)
            _, _, reject, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
            if reject:
                rejections_h0 += 1
        results_h0[i, j] = rejections_h0 / n_trials

        # Synthetic H1: Different distributions
        rejections_h1 = 0
        for _ in range(n_trials):
            X = np.random.normal(0, 1, size=(n_samples, dim))
            Y = np.random.normal(delta, 1, size=(n_samples, dim))
            kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=sigma)
            _, _, reject, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
            if reject:
                rejections_h1 += 1
        results_h1[i, j] = rejections_h1 / n_trials

# Plot results
plt.figure(figsize=(10, 6))
for i, dim in enumerate(dims):
    plt.plot(bandwidths, results_h0[i, :], 'o--', label=f'H0 (dim={dim})', alpha=0.7)
    plt.plot(bandwidths, results_h1[i, :], 'o-', label=f'H1 (dim={dim})', alpha=0.7)

plt.xlabel('Kernel Bandwidth (Sigma)')
plt.ylabel('Probability of Rejections')
plt.title('Effect of Dimensionality and Kernel Bandwidth on MMD Test')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/dim_bandwidth_prob_lineplot.png')
plt.show()