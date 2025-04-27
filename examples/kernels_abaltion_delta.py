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
deltas = np.linspace(0, 1, 11)  # Mean shift values to test
dim = 10  # Fixed dimensionality
n_samples = 50  # Fixed sample size
n_trials = 20
B = 300
alpha = 0.05

# Define kernels
kernels = {
    "Gaussian RBF": lambda X, Y: gaussian_rbf_kernel(X, Y, sigma=median_heuristic(X, Y)),
    "Linear": linear_kernel,
    "Polynomial": lambda X, Y: polynomial_kernel(X, Y, degree=3, coef0=1),
    "Sigmoid": lambda X, Y: sigmoid_kernel(X, Y, alpha=1.0, coef0=0.0),
    "Exponential": lambda X, Y: exponential_kernel(X, Y, sigma=median_heuristic(X, Y)),
}

# Results storage
results = {kernel_name: {"h0": [], "h1": []} for kernel_name in kernels}

# Ablation over delta for all kernels
for kernel_name, kernel_func in kernels.items():
    print(f"Testing kernel: {kernel_name}")
    for delta in deltas:
        h1_rejections = 0
        for _ in range(n_trials):

            # H1: Different distributions
            X = np.random.normal(0, 1, size=(n_samples, dim))
            Y = np.random.normal(delta, 1, size=(n_samples, dim))
            _, _, reject_h1, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
            if reject_h1:
                h1_rejections += 1
        results[kernel_name]["h1"].append(h1_rejections / n_trials)

# Plot results
plt.figure(figsize=(12, 8))
for kernel_name in kernels:
    plt.plot(deltas, results[kernel_name]["h1"], 'o-', label=f'H1 (Power) - {kernel_name}', alpha=0.7)

plt.xlabel('Delta (Mean Shift)')
plt.ylabel('Probability of Rejections')
plt.title('Effect of Delta (Mean Shift) on Kernels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/kernel_ablation_delta_all_kernels.png')
plt.show()