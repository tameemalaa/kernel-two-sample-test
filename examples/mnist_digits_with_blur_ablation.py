from src.utils import load_mnist_digits, median_heuristic
from src.kernel import gaussian_rbf_kernel
from src.permutation import permutation_test
from src.permutation_energy import permutation_test_energy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Parameters
n_total = 100
blur_levels = np.linspace(0, 2, 11)  # Varying blur levels (sigma values)
B = 300
alpha = 0.05

# Results storage
results_mmd = {"mmds": [], "pvals": [], "rejections": []}
results_energy = {"pvals": [], "rejections": []}

# Load MNIST digit "0"
np.random.seed(123)
all_zeros = load_mnist_digits(0, n_samples=n_total)
indices = np.random.permutation(n_total)
X = all_zeros[indices[:n_total // 2]]
Y_original = all_zeros[indices[n_total // 2:]]

# Ablation over blur levels
for blur_sigma in blur_levels:
    blurred_Y = np.array([gaussian_filter(img, sigma=blur_sigma) for img in Y_original])

    # Compute MMD
    bandwidth = median_heuristic(X, blurred_Y)
    kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=bandwidth)
    observed_mmd, p_val_mmd, reject_mmd, _ = permutation_test(
        X, blurred_Y, kernel_func=kernel_func, B=B, alpha=alpha
    )

    # Compute Energy Distance
    _, p_val_energy, reject_energy, _ = permutation_test_energy(
        X, blurred_Y, num_permutations=B, alpha=alpha
    )

    # Store results for MMD
    results_mmd["mmds"].append(observed_mmd)
    results_mmd["pvals"].append(p_val_mmd)
    results_mmd["rejections"].append(reject_mmd)

    # Store results for Energy Distance
    results_energy["pvals"].append(p_val_energy)
    results_energy["rejections"].append(reject_energy)

    print(f"Blur sigma: {blur_sigma:.2f}, MMD^2: {observed_mmd:.5f}, p-value (MMD): {p_val_mmd:.5f}, Reject H0 (MMD): {reject_mmd}")
    print(f"Blur sigma: {blur_sigma:.2f}, p-value (Energy): {p_val_energy:.5f}, Reject H0 (Energy): {reject_energy}")

# Plot results
# Observed MMD^2
plt.figure(figsize=(8, 6))
plt.plot(blur_levels, results_mmd["mmds"], 'o-', label='Observed MMD^2 (MMD)', alpha=0.7)
plt.xlabel('Blur Level (Sigma)')
plt.ylabel('Observed MMD^2')
plt.title('Effect of Blur Level on MMD Test')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/mnist_blur_ablation_mmd.png')
plt.show()

# p-values
plt.figure(figsize=(8, 6))
plt.plot(blur_levels, results_mmd["pvals"], 'o-', label='p-value (MMD)', alpha=0.7)
plt.plot(blur_levels, results_energy["pvals"], 's-', label='p-value (Energy)', alpha=0.7)
plt.axhline(y=alpha, color='r', linestyle='--', label='Significance Level (alpha)')
plt.xlabel('Blur Level (Sigma)')
plt.ylabel('p-value')
plt.title('Effect of Blur Level on p-value (MMD vs. Energy)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/mnist_blur_ablation_pval.png')
plt.show()

# Rejection Rates
plt.figure(figsize=(8, 6))
plt.plot(blur_levels, results_mmd["rejections"], 'o-', label='Reject H0 (MMD)', alpha=0.7)
plt.plot(blur_levels, results_energy["rejections"], 's-', label='Reject H0 (Energy)', alpha=0.7)
plt.xlabel('Blur Level (Sigma)')
plt.ylabel('Rejection Rate')
plt.title('Effect of Blur Level on Rejection of H0 (MMD vs. Energy)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/mnist_blur_ablation_rejection.png')
plt.show()