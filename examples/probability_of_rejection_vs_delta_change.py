import numpy as np
import matplotlib.pyplot as plt
from src.kernel import gaussian_rbf_kernel
from src.permutation import permutation_test
from src.permutation_energy import permutation_test_energy
from src.utils import median_heuristic

# Parameters
np.random.seed(42)
deltas = np.linspace(0, 1, 11)  # Different delta values to test
n_samples = 50
n_trials = 20
B = 300
alpha = 0.05
dim = 10  # Fixed dimensionality

def run_power_curve_delta(generator_X, generator_Y, label, test_func):
    powers = []
    for delta in deltas:
        rejections = 0
        for _ in range(n_trials):
            X = generator_X(dim)
            Y = generator_Y(dim, delta)

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
        print(f"{label} | Delta: {delta:.2f}, {'Type I Error' if label.endswith('H0 TRUE') else 'Power'}: {power:.2f}")
    return powers


# --- Synthetic: H1 true ---
gen_syn_h1 = lambda dim: np.random.normal(0, 1, size=(n_samples, dim))
gen_syn_h1_shifted = lambda dim, delta: np.random.normal(delta, 1, size=(n_samples, dim))
powers_synth_reject_mmd = run_power_curve_delta(gen_syn_h1, gen_syn_h1_shifted, "Synthetic H1 TRUE (MMD)", "mmd")
powers_synth_reject_energy = run_power_curve_delta(gen_syn_h1, gen_syn_h1_shifted, "Synthetic H1 TRUE (Energy)", "energy")

# --- Plotting ---
plt.figure(figsize=(10, 8))
plt.plot(deltas, powers_synth_reject_mmd, 'o-', label='Synthetic H1 TRUE (MMD)', alpha=0.5)
plt.plot(deltas, powers_synth_reject_energy, 'x-', label='Synthetic H1 TRUE (Energy)', alpha=0.5)

plt.xlabel('Delta')
plt.ylabel('Probability of Rejections')
plt.title('Probability of Rejections vs. Delta (MMD vs. Energy)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/probability_of_rejection_vs_delta_mmd_vs_energy.png')
plt.show()