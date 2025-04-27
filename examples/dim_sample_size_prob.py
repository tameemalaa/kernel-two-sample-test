import numpy as np
import matplotlib.pyplot as plt
from src.kernel import gaussian_rbf_kernel
from src.permutation import permutation_test
from src.utils import median_heuristic

# Parameters
np.random.seed(42)
dims = [1,3,5 ]  # Different dimensionalities
sample_sizes = [10, 20,40, 60, 80, 100]  # Different sample sizes
delta = 0.5
n_trials = 10
B = 300
alpha = 0.05

# Results storage
results_h0 = np.zeros((len(dims), len(sample_sizes)))
results_h1 = np.zeros((len(dims), len(sample_sizes)))

# Ablation over dimensionality and sample size
for i, dim in enumerate(dims):
    for j, n_samples in enumerate(sample_sizes):
        # Synthetic H0: Identical distributions
        rejections_h0 = 0
        for _ in range(n_trials):
            X = np.random.normal(0, 1, size=(n_samples, dim))
            Y = np.random.normal(0, 1, size=(n_samples, dim))
            bandwidth = median_heuristic(X, Y)
            kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=bandwidth)
            _, _, reject, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
            if reject:
                rejections_h0 += 1
        results_h0[i, j] = rejections_h0 / n_trials

        # Synthetic H1: Different distributions
        rejections_h1 = 0
        for _ in range(n_trials):
            X = np.random.normal(0, 1, size=(n_samples, dim))
            Y = np.random.normal(delta, 1, size=(n_samples, dim))
            bandwidth = median_heuristic(X, Y)
            kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=bandwidth)
            _, _, reject, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
            if reject:
                rejections_h1 += 1
        results_h1[i, j] = rejections_h1 / n_trials

# Plot results
plt.figure(figsize=(10, 6))
for i, dim in enumerate(dims):
    plt.plot(sample_sizes, results_h0[i, :], 'o--', label=f'H0 (dim={dim})', alpha=0.7)
    plt.plot(sample_sizes, results_h1[i, :], 'o-', label=f'H1 (dim={dim})', alpha=0.7)

plt.xlabel('Sample Size')
plt.ylabel('Probability of Rejections')
plt.title('Effect of Dimensionality and Sample Size on MMD Test')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/ablation_dim_sample_size.png')
plt.show()