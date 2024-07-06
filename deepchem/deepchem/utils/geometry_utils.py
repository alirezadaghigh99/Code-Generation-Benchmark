def compute_centroid(coordinates: np.ndarray) -> np.ndarray:
    """Compute the (x,y,z) centroid of provided coordinates

    Parameters
    ----------
    coordinates: np.ndarray
        A numpy array of shape `(N, 3)`, where `N` is the number of atoms.

    Returns
    -------
    centroid: np.ndarray
        A numpy array of shape `(3,)`, where `3` is (x,y,z).
    """
    centroid = np.mean(coordinates, axis=0)
    return centroid

