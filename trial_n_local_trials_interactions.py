import numpy as np
import matplotlib.pyplot as plt
from kevins_master_algorithm import _kevins_algorithm
from greedy_plus_plus_algorithm import _greedy_plus_plus_algorithm
from scipy.io import loadmat
from pathlib import Path

# Load interactions matrix
script_dir = Path(__file__).resolve().parent
X = loadmat(script_dir / "interactions.mat")["B"]

# Parameters
sizes = np.arange(1, 121)
n_local_trials_values = [10, 20, 30, 40, 50]
n_clusters = sizes.max()
n_repeats = 5
X_norm = np.linalg.norm(X, ord="fro")

# Initiate trial tensors
kevins_scores = np.zeros((len(n_local_trials_values), n_repeats, n_clusters))
greedy_plus_plus_scores = np.zeros_like(kevins_scores)

for j, n_local_trials in enumerate(n_local_trials_values):
    for t in range(n_repeats):
        seed = np.random.randint(1, 100)

        # Compute kevins algorithm residual
        centers_ka, inds_ka, res_ka, _ = _kevins_algorithm(
            X, n_clusters, np.random.RandomState(seed), n_local_trials
        )
        kevins_scores[j, t] = res_ka

        # Compute greedy++ residual
        centers_gpp, inds_gpp, res_gpp, _ = _greedy_plus_plus_algorithm(
            X, n_clusters, np.random.RandomState(seed), n_local_trials
        )
        greedy_plus_plus_scores[j, t] = res_gpp

# Convert to normalized residuals
kevins_scores /= X_norm
greedy_plus_plus_scores /= X_norm

# Calculate mean and standard deviation for each algorithm
kevins_mean = kevins_scores.mean(axis=1)
kevins_std = kevins_scores.std(axis=1)
greedy_plus_plus_mean = greedy_plus_plus_scores.mean(axis=1)
greedy_plus_plus_std = greedy_plus_plus_scores.std(axis=1)

# Plot
plt.figure(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0, 1, len(n_local_trials_values)))

for j, n_local_trials in enumerate(n_local_trials_values):
    plt.semilogy(
        sizes,
        kevins_mean[j],
        label=f"Kevins, n_local_trials={n_local_trials}",
        color=colors[j],
        linestyle='-'
    )
    plt.fill_between(
        sizes,
        kevins_mean[j] - kevins_std[j],
        kevins_mean[j] + kevins_std[j],
        facecolor=colors[j],
        alpha=0.12,
    )

    plt.semilogy(
        sizes,
        greedy_plus_plus_mean[j],
        label=f"Greedy++, n_local_trials={n_local_trials}",
        color=colors[j],
        linestyle='--'
    )
    plt.fill_between(
        sizes,
        greedy_plus_plus_mean[j] - greedy_plus_plus_std[j],
        greedy_plus_plus_mean[j] + greedy_plus_plus_std[j],
        facecolor=colors[j],
        alpha=0.12,
    )

plt.xlabel('Number of Clusters')
plt.ylabel('Normalized Residual Frobenius Norm')
plt.title('Residual vs. Number of Clusters for Different n_local_trials Values')
plt.legend(fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("residuals_n_local_trials_interactions", dpi=300)
plt.show()