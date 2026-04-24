import numpy as np
import time

def _kevins_algorithm(
    X, n_clusters, random_state, n_local_trials=None
):
    """Low-rank initializer using weighted candidate sampling.

    This algorithm samples candidate points with probability proportional to
    the squared norm of the current residual rows, then chooses the best
    candidate based on projected residual reduction.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data matrix.

    n_clusters : int
        Number of centers to choose.

    random_state : RandomState instance
        Random number generator used to sample candidate points.

    n_local_trials : int, default=None
        Number of random candidates to consider at each iteration. If None,
        the value is set to 2 + log(n_clusters).

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The selected centers from X.

    indices : ndarray of shape (n_clusters,)
        The indices of the chosen centers in X.

    residuals : list of float
        Frobenius norm of the residual after each center selection.

    times : list of float
        Elapsed time at each selection step.
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