def generate_noise_matrix_from_trace(
    K,
    trace,
    *,
    max_trace_prob=1.0,
    min_trace_prob=1e-5,
    max_noise_rate=1 - 1e-5,
    min_noise_rate=0.0,
    valid_noise_matrix=True,
    py=None,
    frac_zero_noise_rates=0.0,
    seed=0,
    max_iter=10000,
) -> Optional[np.ndarray]:
    """Generates a ``K x K`` noise matrix ``P(label=k_s|true_label=k_y)`` with
    ``np.sum(np.diagonal(noise_matrix))`` equal to the given `trace`.

    Parameters
    ----------
    K : int
      Creates a noise matrix of shape ``(K, K)``. Implies there are
      K classes for learning with noisy labels.

    trace : float
      Sum of diagonal entries of array of random probabilities returned.

    max_trace_prob : float
      Maximum probability of any entry in the trace of the return matrix.

    min_trace_prob : float
      Minimum probability of any entry in the trace of the return matrix.

    max_noise_rate : float
      Maximum noise_rate (non-diagonal entry) in the returned np.ndarray.

    min_noise_rate : float
      Minimum noise_rate (non-diagonal entry) in the returned np.ndarray.

    valid_noise_matrix : bool, default=True
      If ``True``, returns a matrix having all necessary conditions for
      learning with noisy labels. In particular, ``p(true_label=k)p(label=k) < p(true_label=k,label=k)``
      is satisfied. This requires that ``trace > 1``.

    py : np.ndarray
      An array of shape ``(K,)`` representing the fraction (prior probability) of each true class label, ``P(true_label = k)``.
      This argument is **required** when ``valid_noise_matrix=True``.

    frac_zero_noise_rates : float
      The fraction of the ``n*(n-1)`` noise rates
      that will be set to 0. Note that if you set a high trace, it may be
      impossible to also have a low fraction of zero noise rates without
      forcing all non-1 diagonal values. Instead, when this happens we only
      guarantee to produce a noise matrix with `frac_zero_noise_rates` *or
      higher*. The opposite occurs with a small trace.

    seed : int
      Seeds the random number generator for numpy.

    max_iter : int, default=10000
      The max number of tries to produce a valid matrix before returning ``None``.

    Returns
    -------
    noise_matrix : np.ndarray or None
      An array of shape ``(K, K)`` representing the noise matrix ``P(label=k_s|true_label=k_y)`` with `trace`
      equal to ``np.sum(np.diagonal(noise_matrix))``. This a conditional probability matrix and a
      left stochastic matrix. Returns ``None`` if `max_iter` is exceeded.
    """

    if valid_noise_matrix and trace <= 1:
        raise ValueError(
            "trace = {}. trace > 1 is necessary for a".format(trace)
            + " valid noise matrix to be returned (valid_noise_matrix == True)"
        )

    if valid_noise_matrix and py is None and K > 2:
        raise ValueError(
            "py must be provided (not None) if the input parameter" + " valid_noise_matrix == True"
        )

    if K <= 1:
        raise ValueError("K must be >= 2, but K = {}.".format(K))

    if max_iter < 1:
        return None

    np.random.seed(seed)

    # Special (highly constrained) case with faster solution.
    # Every 2 x 2 noise matrix with trace > 1 is valid because p(y) is not used
    if K == 2:
        if frac_zero_noise_rates >= 0.5:  # Include a single zero noise rate
            noise_mat = np.array(
                [
                    [1.0, 1 - (trace - 1.0)],
                    [0.0, trace - 1.0],
                ]
            )
            return noise_mat if np.random.rand() > 0.5 else np.rot90(noise_mat, k=2)
        else:  # No zero noise rates
            diag = generate_n_rand_probabilities_that_sum_to_m(2, trace)
            noise_matrix = np.array(
                [
                    [diag[0], 1 - diag[1]],
                    [1 - diag[0], diag[1]],
                ]
            )
            return noise_matrix

            # K > 2
    for z in range(max_iter):
        noise_matrix = np.zeros(shape=(K, K))

        # Randomly generate noise_matrix diagonal.
        nm_diagonal = generate_n_rand_probabilities_that_sum_to_m(
            n=K,
            m=trace,
            max_prob=max_trace_prob,
            min_prob=min_trace_prob,
        )
        np.fill_diagonal(noise_matrix, nm_diagonal)

        # Randomly distribute number of zero-noise-rates across columns
        num_col_with_noise = K - np.count_nonzero(1 == nm_diagonal)
        num_zero_noise_rates = int(K * (K - 1) * frac_zero_noise_rates)
        # Remove zeros already in [1,0,..,0] columns
        num_zero_noise_rates -= (K - num_col_with_noise) * (K - 1)
        num_zero_noise_rates = np.maximum(num_zero_noise_rates, 0)  # Prevent negative
        num_zero_noise_rates_per_col = (
            randomly_distribute_N_balls_into_K_bins(
                N=num_zero_noise_rates,
                K=num_col_with_noise,
                max_balls_per_bin=K - 2,
                # 2 = one for diagonal, and one to sum to 1
                min_balls_per_bin=0,
            )
            if K > 2
            else np.array([0, 0])
        )  # Special case when K == 2
        stack_nonzero_noise_rates_per_col = list(K - 1 - num_zero_noise_rates_per_col)[::-1]
        # Randomly generate noise rates for columns with noise.
        for col in np.arange(K)[nm_diagonal != 1]:
            num_noise = stack_nonzero_noise_rates_per_col.pop()
            # Generate num_noise noise_rates for the given column.
            noise_rates_col = list(
                generate_n_rand_probabilities_that_sum_to_m(
                    n=num_noise,
                    m=1 - nm_diagonal[col],
                    max_prob=max_noise_rate,
                    min_prob=min_noise_rate,
                )
            )
            # Randomly select which rows of the noisy column to assign the
            # random noise rates
            rows = np.random.choice(
                [row for row in range(K) if row != col], num_noise, replace=False
            )
            for row in rows:
                noise_matrix[row][col] = noise_rates_col.pop()
        if not valid_noise_matrix or noise_matrix_is_valid(noise_matrix, py):
            return noise_matrix

    return None

def noise_matrix_is_valid(noise_matrix, py, *, verbose=False) -> bool:
    """Given a prior `py` representing ``p(true_label=k)``, checks if the given `noise_matrix` is a
    learnable matrix. Learnability means that it is possible to achieve
    better than random performance, on average, for the amount of noise in
    `noise_matrix`.

    Parameters
    ----------
    noise_matrix : np.ndarray
      An array of shape ``(K, K)`` representing the conditional probability
      matrix ``P(label=k_s|true_label=k_y)`` containing the fraction of
      examples in every class, labeled as every other class. Assumes columns of
      `noise_matrix` sum to 1.

    py : np.ndarray
      An array of shape ``(K,)`` representing the fraction (prior probability)
      of each true class label, ``P(true_label = k)``.

    Returns
    -------
    is_valid : bool
      Whether the noise matrix is a learnable matrix.
    """

    # Number of classes
    K = len(py)

    # let's assume some number of training examples for code readability,
    # but it doesn't matter what we choose as it's not actually used.
    N = float(10000)

    ps = np.dot(noise_matrix, py)  # P(true_label=k)

    # P(label=k, true_label=k')
    joint_noise = np.multiply(noise_matrix, py)  # / float(N)

    # Check that joint_probs is valid probability matrix
    if not (abs(joint_noise.sum() - 1.0) < FLOATING_POINT_COMPARISON):
        return False

    # Check that noise_matrix is a valid matrix
    # i.e. check p(label=k)*p(true_label=k) < p(label=k, true_label=k)
    for i in range(K):
        C = N * joint_noise[i][i]
        E1 = N * joint_noise[i].sum() - C
        E2 = N * joint_noise.T[i].sum() - C
        O = N - E1 - E2 - C
        if verbose:
            print(
                "E1E2/C",
                round(E1 * E2 / C),
                "E1",
                round(E1),
                "E2",
                round(E2),
                "C",
                round(C),
                "|",
                round(E1 * E2 / C + E1 + E2 + C),
                "|",
                round(E1 * E2 / C),
                "<",
                round(O),
            )
            print(
                round(ps[i] * py[i]),
                "<",
                round(joint_noise[i][i]),
                ":",
                ps[i] * py[i] < joint_noise[i][i],
            )

        if not (ps[i] * py[i] < joint_noise[i][i]):
            return False

    return True

