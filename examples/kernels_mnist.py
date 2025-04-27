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
from src.utils import median_heuristic, load_mnist_digits

# Parameters
np.random.seed(42)
n_samples = 50  # Number of samples per group
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

# Results storage
results = {kernel_name: {"h0": [], "h1": []} for kernel_name in kernels}

# --- MNIST Data ---
for kernel_name, kernel_func in kernels.items():
    print(f"Testing kernel: {kernel_name}")
    h0_rejections = 0
    h1_rejections = 0
    for _ in range(n_trials):
        # H0: Split zeros
        all_zeros = load_mnist_digits(0, n_samples=2 * n_samples)
        idx = np.random.permutation(2 * n_samples)
        X = all_zeros[idx[:n_samples]]
        Y = all_zeros[idx[n_samples:]]
        _, _, reject_h0, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
        if reject_h0:
            h0_rejections += 1

        # H1: Zero vs One
        X = load_mnist_digits(0, n_samples=n_samples)
        Y = load_mnist_digits(1, n_samples=n_samples)
        _, _, reject_h1, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
        if reject_h1:
            h1_rejections += 1

    results[kernel_name]["h0"].append(h0_rejections / n_trials)
    results[kernel_name]["h1"].append(h1_rejections / n_trials)

# --- Plotting Results ---
kernel_names = list(kernels.keys())
h0_rejection_rates = [results[kernel]["h0"][0] for kernel in kernel_names]
h1_rejection_rates = [results[kernel]["h1"][0] for kernel in kernel_names]

x = np.arange(len(kernel_names))  # Kernel indices
width = 0.35  # Bar width

plt.figure(figsize=(14, 8))
bars_h0 = plt.bar(x - width / 2, h0_rejection_rates, width, label='H0 (Type I Error)', alpha=0.8, color='skyblue')
bars_h1 = plt.bar(x + width / 2, h1_rejection_rates, width, label='H1 (Power)', alpha=0.8, color='salmon')

# Add data labels
for bar in bars_h0:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{bar.get_height():.2f}', 
             ha='center', va='bottom', fontsize=10)
for bar in bars_h1:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{bar.get_height():.2f}', 
             ha='center', va='bottom', fontsize=10)

# Customize plot
plt.xticks(x, kernel_names, rotation=45, ha='right', fontsize=12)
plt.xlabel('Kernel', fontsize=14)
plt.ylabel('Rejection Rate', fontsize=14)
plt.title('Effect of Kernels on MNIST Data (H0 vs H1)', fontsize=16)
plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show plot
plt.savefig('results/mnist_kernel_ablation.png')
plt.show()