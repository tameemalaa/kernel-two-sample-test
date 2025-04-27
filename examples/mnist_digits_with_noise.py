from src.utils import load_mnist_digits, median_heuristic
from src.kernel import gaussian_rbf_kernel
from src.permutation import permutation_test
from src.visualization import plot_mmd_permutation_test
import numpy as np

np.random.seed(123)
all_zeros = load_mnist_digits(0, n_samples=100)
indices = np.random.permutation(100)
X = all_zeros[indices[:100 // 2]]
Y_orignal = all_zeros[indices[100 // 2:]]

noise = np.random.normal(0, 0.1, X.shape)
Y = Y_orignal + noise

bandwidth = median_heuristic(X, Y)
kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=bandwidth)

observed_mmd, p_val, reject, mmd_permutations = permutation_test(
    X, Y, kernel_func=kernel_func, B=300, alpha=0.05
)
print(f"Observed MMD^2: {observed_mmd:.5f}")
print(f"Permutation test p-value: {p_val:.5f}")
print("Reject H0 (distributions differ):" if reject else "Fail to reject H0 (distributions similar)")

plot_mmd_permutation_test(
    mmd_permutations, observed_mmd,
    title='MNIST 0 vs 0+noise: MMD Permutation Test',
    filename='results/mnist_0_vs_0noise_mmd.png'
)