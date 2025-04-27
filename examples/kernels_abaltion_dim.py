import numpy as np
import matplotlib.pyplot as plt
from src.kernel import (
    gaussian_rbf_kernel,
    linear_kernel,
    polynomial_kernel,
    sigmoid_kernel,
    exponential_kernel,
)
from src.permutation import permutation_test
from src.utils import median_heuristic

# Parameters
np.random.seed(42)
dims = np.arange(1, 12, 1)  # Different dimensionalities to test
n_samples = 50
kernels = {
    "Gaussian RBF": lambda X, Y: gaussian_rbf_kernel(X, Y, sigma=median_heuristic(X, Y)),
    "Linear": linear_kernel,
    "Polynomial": lambda X, Y: polynomial_kernel(X, Y, degree=3, coef0=1),
    "Sigmoid": lambda X, Y: sigmoid_kernel(X, Y, alpha=1.0, coef0=0.0),
    "Exponential": lambda X, Y: exponential_kernel(X, Y, sigma=median_heuristic(X, Y)),
}
n_trials = 20
B = 300
alpha = 0.05
delta = 0.5

# Results storage
results = {kernel_name: {"h0": [], "h1": []} for kernel_name in kernels}

# Ablation over dimensionality
for kernel_name, kernel_func in kernels.items():
    print(f"Testing kernel: {kernel_name}")
    for dim in dims:
        h0_rejections = 0
        h1_rejections = 0
        for _ in range(n_trials):
            # H0: Identical distributions
            X = np.random.normal(0, 1, size=(n_samples, dim))
            Y = np.random.normal(0, 1, size=(n_samples, dim))
            _, _, reject_h0, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
            if reject_h0:
                h0_rejections += 1

            # H1: Different distributions
            X = np.random.normal(0, 1, size=(n_samples, dim))
            Y = np.random.normal(delta, 1, size=(n_samples, dim))
            _, _, reject_h1, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
            if reject_h1:
                h1_rejections += 1

        results[kernel_name]["h0"].append(h0_rejections / n_trials)
        results[kernel_name]["h1"].append(h1_rejections / n_trials)

# Plot results
plt.figure(figsize=(10, 6))
for kernel_name in kernels:
    plt.plot(dims, results[kernel_name]["h0"], 'o--', label=f'H0 (Type I Error) - {kernel_name}', alpha=0.7)
    plt.plot(dims, results[kernel_name]["h1"], 'o-', label=f'H1 (Power) - {kernel_name}', alpha=0.7)
plt.xlabel('Dimensionality')
plt.ylabel('Probability of Rejections')
plt.title('Kernel Ablation: Dimensionality')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/kernel_ablation_dimensionality.png')
plt.show()