import numpy as np
import matplotlib.pyplot as plt
from src.permutation_energy import permutation_test_energy
from src.permutation import permutation_test
from src.kernel import gaussian_rbf_kernel
from src.utils import median_heuristic
np.random.seed(42)
n_samples = 50
dim = 10
deltas = np.linspace(0, 1, 11)
n_trials = 20
alpha = 0.05

powers_mmd = []
powers_energy = []
fprs_mmd = []
fprs_energy = []


for delta in deltas:
    powers_mmd_trial = []
    powers_energy_trial = []
    for _ in range(n_trials):
        # --- Power: H1 (different distributions)
        X = np.random.normal(0, 1, (n_samples, dim))
        Y = np.random.normal(delta, 1, (n_samples, dim))
        bandwidth = median_heuristic(X, Y)
        kernel_func = lambda a,b: gaussian_rbf_kernel(a, b, sigma=bandwidth)
        mmd_stat, pval_mmd, reject_null_mmd, _ = permutation_test(X, Y, kernel_func=kernel_func, B=300)
        powers_mmd_trial.append(reject_null_mmd)
        energy_stat, pval_energy, reject_null_energy, _ = permutation_test_energy(X, Y, num_permutations=300)
        powers_energy_trial.append(reject_null_energy)
        print(f"delta: {delta:.2f}, mmd_stat: {mmd_stat:.4f}, pval_mmd: {pval_mmd:.4f}, reject_null_mmd: {reject_null_mmd}")
        print(f"delta: {delta:.2f}, energy_stat: {energy_stat:.4f}, pval_energy: {pval_energy:.4f}, reject_null_energy: {reject_null_energy}")

    powers_mmd.append(np.mean(powers_mmd_trial))
    powers_energy.append(np.mean(powers_energy_trial))



# Plot results
plt.figure(figsize=(10, 6))

# Plot Power (H1)
plt.plot(deltas, powers_mmd, 'o-', label='Power (H1) - MMD', alpha=0.7)
plt.plot(deltas, powers_energy, 's-', label='Power (H1) - Energy', alpha=0.7)


# Labels and title
plt.xlabel('Mean Shift (Delta)')
plt.ylabel('Power')
plt.title('MMD vs. Energy Distance: Power')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show the plot
plt.savefig('results/mmd_vs_energy_power.png')
plt.show()