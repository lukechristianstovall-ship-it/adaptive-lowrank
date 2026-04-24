import numpy as np

def _random_selection(
    X, n_clusters, random_state
):
    """Random low-rank initializer that selects centers uniformly from X.

    This function chooses n_clusters distinct points at random from X and
    updates the residual by projecting out the selected centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data matrix.

    n_clusters : int
        Number of centers to choose.

    random_state : RandomState instance
        Random number generator used to sample the centers.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The selected centers from X.

    indices : ndarray of shape (n_clusters,)
        The indices of the chosen centers in X.

    residuals : list of float
        Frobenius norm of the residual after each center selection.
    """
    R = X.copy()
    n_samples, n_features = X.shape
    indices = np.full(n_clusters, -1, dtype=int)
    residuals = []

    # Randomly select n_clusters distinct indices
    selected_indices = random_state.choice(n_samples, size=n_clusters, replace=False)
    
    for c, idx in enumerate(selected_indices):
        
        indices[c] = idx
        r_star = R[idx]

        # Update R after adding the new center
        projection = np.outer(R @ r_star, r_star) / (np.linalg.norm(r_star) ** 2)
        R = R - projection

        residuals.append(np.linalg.norm(R, ord='fro'))

    centers= X[indices]
    return centers, indices, residuals