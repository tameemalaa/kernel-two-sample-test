import numpy as np
from src.kernel import gaussian_rbf_kernel
from src.permutation import permutation_test
from src.utils import median_heuristic
from src.visualization import plot_mmd_permutation_test

np.random.seed(42)
n_samples = 100
dim = 10

# Both samples from N(0, I)
X = np.random.normal(0, 1, size=(n_samples, dim))
Y = np.random.normal(0, 1, size=(n_samples, dim))

# Kernel bandwidth via median heuristic
bandwidth = median_heuristic(X, Y)
kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=bandwidth)

# Run permutation test
observed_mmd, p_val, reject, mmd_permutations = permutation_test(
    X, Y, kernel_func=kernel_func, B=300, alpha=0.05
)

print(f"Observed MMD^2: {observed_mmd:.5f}")
print(f"Permutation test p-value: {p_val:.5f}")
print("Reject H0 (distributions differ):" if reject else "Fail to reject H0 (distributions similar)")

# Visualize
plot_mmd_permutation_test(
    mmd_permutations, observed_mmd,
    title='Permutation Test (Same Distribution)',
    filename='results/synth_same_dist.png'
)