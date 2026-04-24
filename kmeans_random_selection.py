import numpy as np

def _random_selection(
    X, n_clusters, random_state
):
    """Random selection of n_clusters centers from X.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The initial data.

    n_clusters : int
        The number of clusters to choose.

    random_state : RandomState instance
        The generator used to initialize the centers.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The randomly selected centers.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X.
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