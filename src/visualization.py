import matplotlib.pyplot as plt

def plot_mmd_permutation_test(mmd_permutations, observed_mmd, title='', filename=None):
    plt.figure(figsize=(7,5))
    plt.hist(mmd_permutations, bins=30, alpha=0.7, color='skyblue', label='Null distribution (permutations)')
    plt.axvline(observed_mmd, color='red', linestyle='--', linewidth=2, label='Observed MMD')
    plt.title(title)
    plt.xlabel('MMD^2')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()