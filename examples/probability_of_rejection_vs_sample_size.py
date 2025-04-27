import numpy as np
import matplotlib.pyplot as plt
from src.kernel import gaussian_rbf_kernel
from src.permutation import permutation_test
from src.permutation_energy import permutation_test_energy
from src.utils import median_heuristic

np.random.seed(42)
sample_sizes = np.arange(10, 110, 10)
n_trials = 20
B = 300
alpha = 0.05
delta = 0.5

def run_power_curve(generator_X, generator_Y, label, test_func):
    powers = []
    for n in sample_sizes:
        rejections = 0
        for _ in range(n_trials):
            X = generator_X(n)
            Y = generator_Y(n)

            if test_func == "mmd":
                bandwidth = median_heuristic(X, Y)
                kernel_func = lambda a, b: gaussian_rbf_kernel(a, b, sigma=bandwidth)
                _, _, reject, _ = permutation_test(X, Y, kernel_func, B=B, alpha=alpha)
            elif test_func == "energy":
                _, _, reject, _ = permutation_test_energy(X, Y, num_permutations=B, alpha=alpha)
            if reject:
                rejections += 1
        power = rejections / n_trials
        powers.append(power)
        print(f"{label} | Sample size: {n}, {'Type I Error' if label.endswith('H0 TRUE') else 'Power'}: {power:.2f}")
    return powers

# --- Synthetic: H0 true ---
def run_for_dim(dim):
    gen_syn = lambda n: np.random.normal(0, 1, size=(n, dim))
    gen_syn2 = lambda n: np.random.normal(delta, 1, size=(n, dim))

    powers_accept_mmd = run_power_curve(gen_syn, gen_syn, f"Synthetic H0 TRUE (MMD, dim={dim})", "mmd")
    powers_accept_energy = run_power_curve(gen_syn, gen_syn, f"Synthetic H0 TRUE (Energy, dim={dim})", "energy")
    powers_reject_mmd = run_power_curve(gen_syn, gen_syn2, f"Synthetic H1 TRUE (MMD, dim={dim})", "mmd")
    powers_reject_energy = run_power_curve(gen_syn, gen_syn2, f"Synthetic H1 TRUE (Energy, dim={dim})", "energy")

    return powers_accept_mmd, powers_accept_energy, powers_reject_mmd, powers_reject_energy

# Run for dim = 2 and dim = 10
powers_dim2 = run_for_dim(2)
powers_dim10 = run_for_dim(10)

# --- Plotting ---
plt.figure(figsize=(12, 8))

# Plot for dim = 2
plt.plot(sample_sizes, powers_dim2[0], 'o--', label='Synthetic H0 TRUE (MMD, dim=2)', alpha=0.5)
plt.plot(sample_sizes, powers_dim2[2], 'o-', label='Synthetic H1 TRUE (MMD, dim=2)', alpha=0.5)
plt.plot(sample_sizes, powers_dim2[1], 'x--', label='Synthetic H0 TRUE (Energy, dim=2)', alpha=0.5)
plt.plot(sample_sizes, powers_dim2[3], 'x-', label='Synthetic H1 TRUE (Energy, dim=2)', alpha=0.5)

# Plot for dim = 10
plt.plot(sample_sizes, powers_dim10[0], 'o--', label='Synthetic H0 TRUE (MMD, dim=10)', alpha=0.5)
plt.plot(sample_sizes, powers_dim10[2], 'o-', label='Synthetic H1 TRUE (MMD, dim=10)', alpha=0.5)
plt.plot(sample_sizes, powers_dim10[1], 'x--', label='Synthetic H0 TRUE (Energy, dim=10)', alpha=0.5)
plt.plot(sample_sizes, powers_dim10[3], 'x-', label='Synthetic H1 TRUE (Energy, dim=10)', alpha=0.5)

plt.xlabel('Sample size per group')
plt.ylabel('Probability of rejections')
plt.title('Probability of Rejections vs. Sample Size (MMD vs. Energy, dim=2 and dim=10)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/probability_of_rejection_vs_sample_size_dim2_dim10.png')
plt.show()