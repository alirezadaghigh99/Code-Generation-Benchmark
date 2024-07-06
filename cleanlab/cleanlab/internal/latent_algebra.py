def compute_py(
    ps, noise_matrix, inverse_noise_matrix, *, py_method="cnt", true_labels_class_counts=None
) -> np.ndarray:
    """Compute ``py := P(true_labels=k)`` from ``ps := P(labels=k)``, `noise_matrix`, and
    `inverse_noise_matrix`.

    This method is ** ROBUST ** when ``py_method = 'cnt'``
    It may work well even when the noise matrices are estimated
    poorly by using the diagonals of the matrices
    instead of all the probabilities in the entire matrix.

    Parameters
    ----------
    ps : np.ndarray
        Array of shape ``(K, )`` or ``(1, K)`` containing the fraction (prior probability) of each observed, noisy label, P(labels = k)

    noise_matrix : np.ndarray
        A conditional probability matrix ( of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    inverse_noise_matrix : np.ndarray of shape (K, K), K = number of classes
        A conditional probability matrix ( of shape ``(K, K)``) of the form ``P(true_label=k_y|label=k_s)`` representing
        the estimated fraction observed examples in each class `k_s`, that are
        mislabeled examples from every other class `k_y`. If ``None``, the
        inverse_noise_matrix will be computed from `pred_probs` and `labels`.
        Assumes columns of `inverse_noise_matrix` sum to 1.

    py_method : str (Options: ["cnt", "eqn", "marginal", "marginal_ps"])
        How to compute the latent prior ``p(true_label=k)``. Default is "cnt" as it often
        works well even when the noise matrices are estimated poorly by using
        the matrix diagonals instead of all the probabilities.

    true_labels_class_counts : np.ndarray
        Array of shape ``(K, )`` or ``(1, K)`` containing the marginal counts of the confident joint
        (like ``cj.sum(axis = 0)``).

    Returns
    -------
    py : np.ndarray
        Array of shape ``(K, )`` or ``(1, K)``.
        The fraction (prior probability) of each TRUE class label, ``P(true_label = k)``."""

    if len(np.shape(ps)) > 2 or (len(np.shape(ps)) == 2 and np.shape(ps)[0] != 1):
        w = "Input parameter np.ndarray ps has shape " + str(np.shape(ps))
        w += ", but shape should be (K, ) or (1, K)"
        warnings.warn(w)

    if py_method == "marginal" and true_labels_class_counts is None:
        msg = (
            'py_method == "marginal" requires true_labels_class_counts, '
            "but true_labels_class_counts is None. "
        )
        msg += " Provide parameter true_labels_class_counts."
        raise ValueError(msg)

    if py_method == "cnt":
        # Computing py this way avoids dividing by zero noise rates.
        # More robust bc error est_p(true_label|labels) / est_p(labels|y) ~ p(true_label|labels) / p(labels|y)
        py = (
            inverse_noise_matrix.diagonal()
            / np.clip(noise_matrix.diagonal(), a_min=TINY_VALUE, a_max=None)
            * ps
        )
        # Equivalently: py = (true_labels_class_counts / labels_class_counts) * ps
    elif py_method == "eqn":
        py = np.linalg.inv(noise_matrix).dot(ps)
    elif py_method == "marginal":
        py = true_labels_class_counts / np.clip(
            float(sum(true_labels_class_counts)), a_min=TINY_VALUE, a_max=None
        )
    elif py_method == "marginal_ps":
        py = np.dot(inverse_noise_matrix, ps)
    else:
        err = "py_method {}".format(py_method)
        err += " should be in [cnt, eqn, marginal, marginal_ps]"
        raise ValueError(err)

    # Clip py (0,1), s.t. no class should have prob 0, hence 1e-6
    py = clip_values(py, low=CLIPPING_LOWER_BOUND, high=1.0, new_sum=1.0)
    return py

def compute_ps_py_inv_noise_matrix(
    labels, noise_matrix
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``ps := P(labels=k), py := P(true_labels=k)``, and the inverse noise matrix.

    Parameters
    ----------
    labels : np.ndarray
          A discrete vector of noisy labels, i.e. some labels may be erroneous.
          *Format requirements*: for dataset with `K` classes, labels must be in ``{0,1,...,K-1}``.

    noise_matrix : np.ndarray
        A conditional probability matrix (of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1."""

    ps = value_counts(labels) / float(len(labels))  # p(labels=k)
    py, inverse_noise_matrix = compute_py_inv_noise_matrix(ps, noise_matrix)
    return ps, py, inverse_noise_matrix

def compute_pyx(pred_probs, noise_matrix, inverse_noise_matrix):
    """Compute ``pyx := P(true_label=k|x)`` from ``pred_probs := P(label=k|x)``, `noise_matrix` and
    `inverse_noise_matrix`.

    This method is ROBUST - meaning it works well even when the
    noise matrices are estimated poorly by only using the diagonals of the
    matrices which tend to be easy to estimate correctly.

    Parameters
    ----------
    pred_probs : np.ndarray
        ``P(label=k|x)`` is a ``(N x K)`` matrix with K model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation.

    noise_matrix : np.ndarray
        A conditional probability matrix (of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of `noise_matrix` sum to 1.

    inverse_noise_matrix : np.ndarray
        A conditional probability matrix (of shape ``(K, K)``)  of the form ``P(true_label=k_y|label=k_s)`` representing
        the estimated fraction observed examples in each class `k_s`, that are
        mislabeled examples from every other class `k_y`. If None, the
        inverse_noise_matrix will be computed from `pred_probs` and `labels`.
        Assumes columns of `inverse_noise_matrix` sum to 1.

    Returns
    -------
    pyx : np.ndarray
        ``P(true_label=k|x)`` is a  ``(N, K)`` matrix of model-predicted probabilities.
        Each row of this matrix corresponds to an example `x` and contains the model-predicted
        probabilities that `x` belongs to each possible class.
        The columns must be ordered such that these probabilities correspond to class 0,1,2,...
        `pred_probs` should have been computed using 3 (or higher) fold cross-validation."""

    if len(np.shape(pred_probs)) != 2:
        raise ValueError(
            "Input parameter np.ndarray 'pred_probs' has shape "
            + str(np.shape(pred_probs))
            + ", but shape should be (N, K)"
        )

    pyx = (
        pred_probs
        * inverse_noise_matrix.diagonal()
        / np.clip(noise_matrix.diagonal(), a_min=TINY_VALUE, a_max=None)
    )
    # Make sure valid probabilities that sum to 1.0
    return np.apply_along_axis(
        func1d=clip_values, axis=1, arr=pyx, **{"low": 0.0, "high": 1.0, "new_sum": 1.0}
    )

