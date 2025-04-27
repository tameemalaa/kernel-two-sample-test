import numpy as np
import matplotlib.pyplot as plt
from src.kernel import gaussian_rbf_kernel
from src.permutation import permutation_test
from src.utils import load_mnist_digits

# Parameters
np.random.seed(42)
dim = 2
n_samples = 50
bandwidths = np.linspace(0.1, 2.1, 20)  # Varying kernel bandwidths
B = 300
alpha = 0.05
delta = 0.5

# Results storage
results = {
    "Synthetic H0": {"mmds": [], "pvals": [], "rejections": []},
    "Synthetic H1": {"mmds": [], "pvals": [], "rejections": []},
    "MNIST H0": {"mmds": [], "pvals": [], "rejections": []},
    "MNIST H1": {"mmds": [], "pvals": [], "rejections": []},
}

# Synthetic H0: Identical distributions
gen_syn = lambda: np.random.normal(0, 1, size=(n_samples, dim))
for sigma in bandwidths:
    X = gen_syn()
    Y = gen_syn()
    kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=sigma)
    mmd_stat, p_val, reject, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
    results["Synthetic H0"]["mmds"].append(mmd_stat)
    results["Synthetic H0"]["pvals"].append(p_val)
    results["Synthetic H0"]["rejections"].append(reject)

# Synthetic H1: Different distributions
gen_syn2 = lambda: np.random.normal(delta, 1, size=(n_samples, dim))
for sigma in bandwidths:
    X = gen_syn()
    Y = gen_syn2()
    kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=sigma)
    mmd_stat, p_val, reject, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
    results["Synthetic H1"]["mmds"].append(mmd_stat)
    results["Synthetic H1"]["pvals"].append(p_val)
    results["Synthetic H1"]["rejections"].append(reject)

# MNIST H0: Split zeros
def mnist_split_zeros():
    all_zeros = load_mnist_digits(0, n_samples=2 * n_samples)
    idx = np.random.permutation(2 * n_samples)
    return all_zeros[idx[:n_samples]], all_zeros[idx[n_samples:]]

for sigma in bandwidths:
    X, Y = mnist_split_zeros()
    kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=sigma)
    mmd_stat, p_val, reject, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
    results["MNIST H0"]["mmds"].append(mmd_stat)
    results["MNIST H0"]["pvals"].append(p_val)
    results["MNIST H0"]["rejections"].append(reject)

# MNIST H1: Zero vs One
def mnist_zeros(): return load_mnist_digits(0, n_samples=n_samples)
def mnist_ones(): return load_mnist_digits(1, n_samples=n_samples)

for sigma in bandwidths:
    X = mnist_zeros()
    Y = mnist_ones()
    kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=sigma)
    mmd_stat, p_val, reject, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
    results["MNIST H1"]["mmds"].append(mmd_stat)
    results["MNIST H1"]["pvals"].append(p_val)
    results["MNIST H1"]["rejections"].append(reject)

# Plot results
for case in results:
    plt.figure(figsize=(8, 6))
    plt.plot(bandwidths, results[case]["mmds"], 'o-', label='Observed MMD^2')
    plt.xlabel('Kernel Bandwidth (Sigma)')
    plt.ylabel('Observed MMD^2')
    plt.title(f'Effect of Kernel Bandwidth on MMD Test ({case})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/kernel_bandwidth_ablation_mmd_{case.replace(" ", "_").lower()}.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(bandwidths, results[case]["pvals"], 'o-', label='p-value')
    plt.axhline(y=alpha, color='r', linestyle='--', label='Significance Level (alpha)')
    plt.xlabel('Kernel Bandwidth (Sigma)')
    plt.ylabel('p-value')
    plt.title(f'Effect of Kernel Bandwidth on p-value ({case})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/kernel_bandwidth_ablation_pval_{case.replace(" ", "_").lower()}.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(bandwidths, results[case]["rejections"], 'o-', label='Reject H0 (1=True, 0=False)')
    plt.xlabel('Kernel Bandwidth (Sigma)')
    plt.ylabel('Rejection Rate')
    plt.title(f'Effect of Kernel Bandwidth on Rejection of H0 ({case})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/kernel_bandwidth_ablation_rejection_{case.replace(" ", "_").lower()}.png')
    plt.show()