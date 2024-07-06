def _modified_weiszfeld_step(X, x_old):
    """Modified Weiszfeld step.

    This function defines one iteration step in order to approximate the
    spatial median (L1 median). It is a form of an iteratively re-weighted
    least squares method.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    x_old : ndarray of shape = (n_features,)
        Current start vector.

    Returns
    -------
    x_new : ndarray of shape (n_features,)
        New iteration step.

    References
    ----------
    - On Computation of Spatial Median for Robust Data Mining, 2005
      T. Kärkkäinen and S. Äyrämö
      http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf
    """
    diff = X - x_old
    diff_norm = np.sqrt(np.sum(diff**2, axis=1))
    mask = diff_norm >= _EPSILON
    # x_old equals one of our samples
    is_x_old_in_X = int(mask.sum() < X.shape[0])

    diff = diff[mask]
    diff_norm = diff_norm[mask][:, np.newaxis]
    quotient_norm = linalg.norm(np.sum(diff / diff_norm, axis=0))

    if quotient_norm > _EPSILON:  # to avoid division by zero
        new_direction = np.sum(X[mask, :] / diff_norm, axis=0) / np.sum(
            1 / diff_norm, axis=0
        )
    else:
        new_direction = 1.0
        quotient_norm = 1.0

    return (
        max(0.0, 1.0 - is_x_old_in_X / quotient_norm) * new_direction
        + min(1.0, is_x_old_in_X / quotient_norm) * x_old
    )

def _spatial_median(X, max_iter=300, tol=1.0e-3):
    """Spatial median (L1 median).

    The spatial median is member of a class of so-called M-estimators which
    are defined by an optimization problem. Given a number of p points in an
    n-dimensional space, the point x minimizing the sum of all distances to the
    p other points is called spatial median.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    max_iter : int, default=300
        Maximum number of iterations.

    tol : float, default=1.e-3
        Stop the algorithm if spatial_median has converged.

    Returns
    -------
    spatial_median : ndarray of shape = (n_features,)
        Spatial median.

    n_iter : int
        Number of iterations needed.

    References
    ----------
    - On Computation of Spatial Median for Robust Data Mining, 2005
      T. Kärkkäinen and S. Äyrämö
      http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf
    """
    if X.shape[1] == 1:
        return 1, np.median(X.ravel(), keepdims=True)

    tol **= 2  # We are computing the tol on the squared norm
    spatial_median_old = np.mean(X, axis=0)

    for n_iter in range(max_iter):
        spatial_median = _modified_weiszfeld_step(X, spatial_median_old)
        if np.sum((spatial_median_old - spatial_median) ** 2) < tol:
            break
        else:
            spatial_median_old = spatial_median
    else:
        warnings.warn(
            "Maximum number of iterations {max_iter} reached in "
            "spatial median for TheilSen regressor."
            "".format(max_iter=max_iter),
            ConvergenceWarning,
        )
    return n_iter, spatial_median

