import numpy as np
import matplotlib.pyplot as plt
from kmeans_random_selection import _random_selection
from kevins_master_algorithm import _kevins_algorithm
from kmeans_adaptive_sampling import _kmeans_adaptive_sampling
from greedy_algorithm import _greedy_algorithm
from greedy_plus_plus_algorithm import _greedy_plus_plus_algorithm
from scipy.io import loadmat
from pathlib import Path

# Load interactions matrix
script_dir = Path(__file__).resolve().parent
X = loadmat(script_dir / "interactions.mat")["B"]

# Parameters
sizes = np.arange(1, 121)
n_local_trials = None
n_clusters = sizes.max()
n_repeats = 10
X_norm = np.linalg.norm(X, ord="fro")

# Initiate trial matrices
kevins_mat = np.zeros((n_repeats, n_clusters))
adaptive_mat = np.zeros_like(kevins_mat)
random_mat = np.zeros_like(kevins_mat)
greedy_mat = np.zeros_like(kevins_mat)
greedy_plus_plus_mat = np.zeros_like(kevins_mat)
optimal_vec = np.zeros(n_clusters)

# Compute optimal low-rank residual with SVD
U, s, Vt = np.linalg.svd(X, full_matrices=False)
for idx, i in enumerate(sizes):
    optimal_vec[idx] = np.sqrt(np.sum(s[i:]**2)) / X_norm

for t in range(n_repeats):
    seed = np.random.randint(1, 100)

    # Compute kevins algorithm residual
    centers_ka, inds_ka, res_ka, _ = _kevins_algorithm(
        X, n_clusters, np.random.RandomState(seed), n_local_trials
    )
    kevins_mat[t] = res_ka
    
    # Compute adaptive sampling residual
    centers_as, inds_as, res_as, _ = _kmeans_adaptive_sampling(
        X, n_clusters, np.random.RandomState(seed)
    )
    adaptive_mat[t] = res_as

    # Compute random residual
    centers_rand, inds_rand, res_rand = _random_selection(
        X, n_clusters, np.random.RandomState(seed)
    )
    random_mat[t] = res_rand

    # Compute greedy residual
    centers_g, inds_g, res_g, _ = _greedy_algorithm(
        X, n_clusters
    )
    greedy_mat[t] = res_g

    # Compute greedy++ residual
    centers_gpp, inds_gpp, res_gpp, _ = _greedy_plus_plus_algorithm(
        X, n_clusters, np.random.RandomState(seed), n_local_trials
    )
    greedy_plus_plus_mat[t] = res_gpp

# Convert to residuals to residual/norm
kevins_score = kevins_mat / X_norm
adaptive_score = adaptive_mat / X_norm
random_score = random_mat / X_norm
greedy_score = greedy_mat / X_norm
greedy_plus_plus_score = greedy_plus_plus_mat / X_norm

# Calculate mean and standard deviation for each algorithm
kevins_mean = kevins_score.mean(axis=0)
kevins_std = kevins_score.std(axis=0)
adaptive_mean = adaptive_score.mean(axis=0)
adaptive_std = adaptive_score.std(axis=0)
random_mean = random_score.mean(axis=0)
random_std = random_score.std(axis=0)
greedy_mean = greedy_score.mean(axis=0)
greedy_std = greedy_score.std(axis=0)
greedy_plus_plus_mean = greedy_plus_plus_score.mean(axis=0)
greedy_plus_plus_std = greedy_plus_plus_score.std(axis=0)

# Plot
plt.figure(figsize=(10, 6))

def plot_with_error(x, mean, std, label, color):
    plt.semilogy(x, mean, label=label, color=color)
    plt.fill_between(x, mean - std, mean + std,
                     facecolor=color, alpha=0.2)

plot_with_error(sizes, adaptive_mean, adaptive_std,
                'Adaptive sampling', 'tab:orange')
plot_with_error(sizes, kevins_mean, kevins_std,
                "Kevins Algorithm", 'tab:green')
plot_with_error(sizes, greedy_mean, greedy_std,
                'Greedy', 'tab:blue')
plot_with_error(sizes, greedy_plus_plus_mean, greedy_plus_plus_std,
                'Greedy++', 'tab:brown')
plot_with_error(sizes, random_mean, random_std,
                'Random selection', 'tab:red')
plt.semilogy(sizes, optimal_vec, label='Optimal low-rank',
             color='tab:purple')

plt.xlabel('Number of Clusters')
plt.ylabel('Normalized Residual Frobenius Norm')
plt.title('Residual vs. Number of Clusters')
plt.tight_layout()
plt.legend()
plt.savefig("residuals_interactions_trial", dpi=300)
plt.show()