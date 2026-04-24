import numpy as np
import time

def _greedy_algorithm(
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
    n_samples, n_features = X.shape
    indices = np.full(n_clusters, -1, dtype=int)

    # Residuals
    residuals = []

    # Times
    times = []
    start = time.perf_counter()

    # Pick n_clusters points
    for c in range(0, n_clusters):
        # Decide which candidate is best
        RR = R @ R.T
        numerators = np.linalg.norm(RR, axis=0)**2.  
        denominators = np.linalg.norm(R, axis=1)**2.
        mask = ~np.isclose(denominators, 0.0)
        scores = np.zeros_like(numerators)
        scores[mask] = numerators[mask] / denominators[mask]
        best_idx = np.argmax(scores)
        indices[c] = best_idx
        r_star = R[best_idx]
        norm_r_star_sq = np.dot(r_star, r_star)
        projection = np.outer(R @ r_star, r_star) / norm_r_star_sq
        R = R - projection

        end = time.perf_counter()
        times.append(end - start)
        residuals.append(np.linalg.norm(R, ord='fro'))

    centers = X[indices]    
    return centers, indices, residuals, times