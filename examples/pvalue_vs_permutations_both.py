import numpy as np
import matplotlib.pyplot as plt
from src.kernel import gaussian_rbf_kernel
from src.permutation import permutation_test
from src.permutation_energy import permutation_test_energy
from src.utils import median_heuristic, load_mnist_digits

# Parameters
np.random.seed(42)
n_samples = 50
dim = 2
delta = 0.5
B_list = [20, 50, 100, 200]  # Number of permutations
n_trials = 20
alpha = 0.05

def single_case_pvals(gen_X, gen_Y, label, test_func):
    mean_pvals, std_pvals = [], []
    for B in B_list:
        pvals = []
        for _ in range(n_trials):
            X = gen_X()
            Y = gen_Y()
            if test_func == "mmd":
                bandwidth = median_heuristic(X, Y)
                kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=bandwidth)
                _, p_val, _, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
            elif test_func == "energy":
                _, p_val, _, _ = permutation_test_energy(X, Y, num_permutations=B, alpha=alpha)
            pvals.append(p_val)
        mean_pvals.append(np.mean(pvals))
        std_pvals.append(np.std(pvals))
        print(f"{label} | B={B}, mean p-value: {np.mean(pvals):.4f}, std: {np.std(pvals):.4f}")
    return mean_pvals, std_pvals

# Synthetic H0 (same distribution)
gen_syn = lambda: np.random.normal(0, 1, size=(n_samples, dim))
meanp_synth_accept_mmd, stdp_synth_accept_mmd = single_case_pvals(gen_syn, gen_syn, "Synthetic H0 TRUE (MMD)", "mmd")
meanp_synth_accept_energy, stdp_synth_accept_energy = single_case_pvals(gen_syn, gen_syn, "Synthetic H0 TRUE (Energy)", "energy")

# Synthetic H1 (different distributions)
gen_syn2 = lambda: np.random.normal(delta, 1, size=(n_samples, dim))
meanp_synth_reject_mmd, stdp_synth_reject_mmd = single_case_pvals(gen_syn, gen_syn2, "Synthetic H1 TRUE (MMD)", "mmd")
meanp_synth_reject_energy, stdp_synth_reject_energy = single_case_pvals(gen_syn, gen_syn2, "Synthetic H1 TRUE (Energy)", "energy")

# MNIST H0 (split zeros)
def mnist_split():
    all_zeros = load_mnist_digits(0, n_samples=2 * n_samples)
    idx = np.random.permutation(2 * n_samples)
    return all_zeros[idx[:n_samples]], all_zeros[idx[n_samples:]]

meanp_mnist_accept_mmd, stdp_mnist_accept_mmd = [], []
meanp_mnist_accept_energy, stdp_mnist_accept_energy = [], []
for B in B_list:
    pvals_mmd, pvals_energy = [], []
    for _ in range(n_trials):
        X, Y = mnist_split()
        bandwidth = median_heuristic(X, Y)
        kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=bandwidth)
        _, p_val_mmd, _, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
        _, p_val_energy, _, _ = permutation_test_energy(X, Y, num_permutations=B, alpha=alpha)
        pvals_mmd.append(p_val_mmd)
        pvals_energy.append(p_val_energy)
    meanp_mnist_accept_mmd.append(np.mean(pvals_mmd))
    stdp_mnist_accept_mmd.append(np.std(pvals_mmd))
    meanp_mnist_accept_energy.append(np.mean(pvals_energy))
    stdp_mnist_accept_energy.append(np.std(pvals_energy))
    print(f"MNIST H0 TRUE | B={B}, mean p-value (MMD): {np.mean(pvals_mmd):.4f}, std: {np.std(pvals_mmd):.4f}")
    print(f"MNIST H0 TRUE | B={B}, mean p-value (Energy): {np.mean(pvals_energy):.4f}, std: {np.std(pvals_energy):.4f}")

# MNIST H1 (0 vs 1)
def mnist_zeros(): return load_mnist_digits(0, n_samples=n_samples)
def mnist_ones(): return load_mnist_digits(1, n_samples=n_samples)
meanp_mnist_reject_mmd, stdp_mnist_reject_mmd = single_case_pvals(mnist_zeros, mnist_ones, "MNIST H1 TRUE (MMD)", "mmd")
meanp_mnist_reject_energy, stdp_mnist_reject_energy = single_case_pvals(mnist_zeros, mnist_ones, "MNIST H1 TRUE (Energy)", "energy")

# Plotting
plt.figure(figsize=(10, 6))
# Synthetic H0
plt.errorbar(B_list, meanp_synth_accept_mmd, yerr=stdp_synth_accept_mmd, fmt='o--', label='Synthetic H0 TRUE (MMD)', alpha=0.7)
plt.errorbar(B_list, meanp_synth_accept_energy, yerr=stdp_synth_accept_energy, fmt='x--', label='Synthetic H0 TRUE (Energy)', alpha=0.7)
# Synthetic H1
plt.errorbar(B_list, meanp_synth_reject_mmd, yerr=stdp_synth_reject_mmd, fmt='o-', label='Synthetic H1 TRUE (MMD)', alpha=0.7)
plt.errorbar(B_list, meanp_synth_reject_energy, yerr=stdp_synth_reject_energy, fmt='x-', label='Synthetic H1 TRUE (Energy)', alpha=0.7)
# MNIST H0
plt.errorbar(B_list, meanp_mnist_accept_mmd, yerr=stdp_mnist_accept_mmd, fmt='s--', label='MNIST H0 TRUE (MMD)', alpha=0.7)
plt.errorbar(B_list, meanp_mnist_accept_energy, yerr=stdp_mnist_accept_energy, fmt='d--', label='MNIST H0 TRUE (Energy)', alpha=0.7)
# MNIST H1
plt.errorbar(B_list, meanp_mnist_reject_mmd, yerr=stdp_mnist_reject_mmd, fmt='s-', label='MNIST H1 TRUE (MMD)', alpha=0.7)
plt.errorbar(B_list, meanp_mnist_reject_energy, yerr=stdp_mnist_reject_energy, fmt='d-', label='MNIST H1 TRUE (Energy)', alpha=0.7)

plt.xlabel('Number of Permutations (B)')
plt.ylabel('Mean p-value (± std)')
plt.title('p-value Stability vs. #Permutations (MMD vs. Energy)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/pvalue_vs_permutations_mmd_energy.png')
plt.show()