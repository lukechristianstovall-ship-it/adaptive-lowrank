import numpy as np
import time

def _kevins_algorithm(
    X, n_clusters, random_state, n_local_trials=None
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

    # Set the number of local seeding trials if None is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    start = time.perf_counter()

    # Pick n_clusters points
    for c in range(0, n_clusters):

        # Choose center candidates by sampling with probability proportional to the squared row norms of R
        row_norms_sq = np.sum(R**2, axis=1)
        cum_row_norms = np.cumsum(row_norms_sq)
        rand_vals = random_state.uniform(size=n_local_trials) * cum_row_norms[-1]
        candidate_ids = np.searchsorted(cum_row_norms, rand_vals)
        
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, 0, n_samples - 1, out=candidate_ids)

        # Decide which candidate is best
        RR = R @ R[candidate_ids].T
        numerators = np.linalg.norm(RR, axis=0)**2.  
        denominators = np.linalg.norm(R[candidate_ids], axis=1)**2.
        scores = numerators / denominators
        best_idx = np.argmax(scores)
        best_candidate = candidate_ids[best_idx]
        indices[c] = best_candidate
        r_star = R[best_candidate]
        
        # Update R after adding the new center
        projection = np.outer(R @ r_star, r_star) / (np.linalg.norm(r_star) ** 2)
        R = R - projection

        end = time.perf_counter()
        times.append(end - start)
        
        residuals.append(np.linalg.norm(R, ord='fro'))

    centers = X[indices]    
    return centers, indices, residuals, times