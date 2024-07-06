def _set_order(X, y, order="C"):
    """Change the order of X and y if necessary.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Target values.

    order : {None, 'C', 'F'}
        If 'C', dense arrays are returned as C-ordered, sparse matrices in csr
        format. If 'F', dense arrays are return as F-ordered, sparse matrices
        in csc format.

    Returns
    -------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data with guaranteed order.

    y : ndarray of shape (n_samples,)
        Target values with guaranteed order.
    """
    if order not in [None, "C", "F"]:
        raise ValueError(
            "Unknown value for order. Got {} instead of None, 'C' or 'F'.".format(order)
        )
    sparse_X = sparse.issparse(X)
    sparse_y = sparse.issparse(y)
    if order is not None:
        sparse_format = "csc" if order == "F" else "csr"
        if sparse_X:
            X = X.asformat(sparse_format, copy=False)
        else:
            X = np.asarray(X, order=order)
        if sparse_y:
            y = y.asformat(sparse_format)
        else:
            y = np.asarray(y, order=order)
    return X, y

