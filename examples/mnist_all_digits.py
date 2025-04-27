import numpy as np
from src.utils import load_all_mnist_digits, median_heuristic
from src.kernel import gaussian_rbf_kernel
from src.permutation import permutation_test
from src.visualization import plot_mmd_permutation_test

# Load all MNIST digits and shuffle
n_total = 100  # Adjust the number of samples to load
all_digits = load_all_mnist_digits(n_samples=n_total)  # A function that loads all MNIST digits

# Shuffle and split into two halves
np.random.seed(123)
indices = np.random.permutation(n_total)
X = all_digits[indices[:n_total // 2]]
Y = all_digits[indices[n_total // 2:]]

# Compute kernel bandwidth using median heuristic
bandwidth = median_heuristic(X, Y)
kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=bandwidth)

# Run permutation test
observed_mmd, p_val, reject, mmd_permutations = permutation_test(
    X, Y, kernel_func=kernel_func, B=300, alpha=0.05
)

print(f"Observed MMD^2: {observed_mmd:.5f}")
print(f"Permutation test p-value: {p_val:.5f}")
print("Reject H0 (distributions differ):" if reject else "Fail to reject H0 (distributions similar)")

# Visualize results
plot_mmd_permutation_test(
    mmd_permutations, observed_mmd,
    title='MNIST shuffled: MMD Permutation Test',
    filename='results/mnist_shuffled_mmd.png'
)