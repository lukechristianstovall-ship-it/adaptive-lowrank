import numpy as np
import time

def _kmeans_adaptive_sampling(
    X, n_clusters, random_state
):
    """Low-rank matrix version of kmeans_plusplus intializer.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The intial data.

    n_clusters : int
        The number of seeds to choose.

    random_state : RandomState instance
        The generator used to initialize the centers.

    n_local_trials : int, default=None
        The number of seeding trials for each center,
        of which the one reducing the residual the most is chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)).

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    R = X.copy()
    indices = np.full(n_clusters, -1, dtype=int)

    # Residuals
    residuals = []

    # Times
    times = []

    start = time.perf_counter()

    # Pick n_clusters points
    for c in range(0, n_clusters):
        
        # Choose center candidate by sampling with probability proportional to the squared row norms of R
        row_norms_sq = np.sum(R**2, axis=1)
        cum_row_norms = np.cumsum(row_norms_sq)
        rand_vals = random_state.uniform(size=1) * cum_row_norms[-1]
        center_id = np.searchsorted(cum_row_norms, rand_vals)[0]

        indices[c] = center_id
        r_star = R[center_id]

        # Update R after adding the new center
        projection = np.outer(R @ r_star, r_star) / (np.linalg.norm(r_star) ** 2)
        R = R - projection
        
        end = time.perf_counter()
        times.append(end - start)
        residuals.append(np.linalg.norm(R, ord='fro'))

    centers = X[indices]
    return centers, indices, residuals, times