def probs_to_unitary(probs):
    r"""Calculates the unitary matrix corresponding to an input probability distribution.

    For a given distribution :math:`p(i)`, this function returns the unitary :math:`\mathcal{A}`
    that transforms the :math:`|0\rangle` state as

    .. math::

        \mathcal{A} |0\rangle = \sum_{i} \sqrt{p(i)} |i\rangle,

    so that measuring the resulting state in the computational basis will give the state
    :math:`|i\rangle` with probability :math:`p(i)`. Note that the returned unitary matrix is
    real and hence an orthogonal matrix.

    Args:
        probs (array): input probability distribution as a flat array

    Returns:
        array: unitary

    Raises:
        ValueError: if the input array is not flat or does not correspond to a probability
            distribution

    **Example:**

    >>> p = np.ones(4) / 4
    >>> probs_to_unitary(p)
    array([[ 0.5       ,  0.5       ,  0.5       ,  0.5       ],
           [ 0.5       , -0.83333333,  0.16666667,  0.16666667],
           [ 0.5       ,  0.16666667, -0.83333333,  0.16666667],
           [ 0.5       ,  0.16666667,  0.16666667, -0.83333333]])
    """

    if not qml.math.is_abstract(
        sum(probs)
    ):  # skip check and error if jitting to avoid JAX tracer errors
        if not qml.math.allclose(sum(probs), 1) or min(probs) < 0:
            raise ValueError(
                "A valid probability distribution of non-negative numbers that sum to one "
                "must be input"
            )

    # Using the approach discussed here:
    # https://quantumcomputing.stackexchange.com/questions/10239/how-can-i-fill-a-unitary-knowing-only-its-first-column
    psi = qml.math.sqrt(probs)
    overlap = psi[0]
    denominator = qml.math.sqrt(2 + 2 * overlap)
    psi = qml.math.set_index(psi, 0, psi[0] + 1)  # psi[0] += 1, but JAX-JIT compatible
    psi /= denominator

    dim = len(probs)
    return 2 * qml.math.outer(psi, psi) - np.eye(dim)

def func_to_unitary(func, M):
    r"""Calculates the unitary that encodes a function onto an ancilla qubit register.

    Consider a function defined on the set of integers :math:`X = \{0, 1, \ldots, M - 1\}` whose
    output is bounded in the interval :math:`[0, 1]`, i.e., :math:`f: X \rightarrow [0, 1]`.

    The ``func_to_unitary`` function returns a unitary :math:`\mathcal{R}` that performs the
    transformation:

    .. math::

        \mathcal{R} |i\rangle \otimes |0\rangle = |i\rangle\otimes \left(\sqrt{1 - f(i)}|0\rangle +
        \sqrt{f(i)} |1\rangle\right).

    In other words, for a given input state :math:`|i\rangle \otimes |0\rangle`, this unitary
    encodes the amplitude :math:`\sqrt{f(i)}` onto the :math:`|1\rangle` state of the ancilla qubit.
    Hence, measuring the ancilla qubit will result in the :math:`|1\rangle` state with probability
    :math:`f(i)`.

    Args:
        func (callable): a function defined on the set of integers
            :math:`X = \{0, 1, \ldots, M - 1\}` with output value inside :math:`[0, 1]`
        M (int): the number of integers that the function is defined on

    Returns:
        array: the :math:`\mathcal{R}` unitary

    Raises:
        ValueError: if func is not bounded with :math:`[0, 1]` for all :math:`X`

    **Example:**

    >>> func = lambda i: np.sin(i) ** 2
    >>> M = 16
    >>> func_to_unitary(func, M)
    array([[ 1.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        , -1.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.54030231, ...,  0.        ,
             0.        ,  0.        ],
           ...,
           [ 0.        ,  0.        ,  0.        , ..., -0.13673722,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.75968791,  0.65028784],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.65028784, -0.75968791]])
    """
    unitary = np.zeros((2 * M, 2 * M))

    fs = [func(i) for i in range(M)]
    if not qml.math.is_abstract(
        fs[0]
    ):  # skip check and error if jitting to avoid JAX tracer errors
        if min(fs) < 0 or max(fs) > 1:
            raise ValueError(
                "func must be bounded within the interval [0, 1] for the range of input values"
            )

    for i, f in enumerate(fs):
        # array = set_index(array, idx, val) is a JAX-JIT compatible version of array[idx] = val
        unitary = qml.math.set_index(unitary, (2 * i, 2 * i), qml.math.sqrt(1 - f))
        unitary = qml.math.set_index(unitary, (2 * i + 1, 2 * i), qml.math.sqrt(f))
        unitary = qml.math.set_index(unitary, (2 * i, 2 * i + 1), qml.math.sqrt(f))
        unitary = qml.math.set_index(unitary, (2 * i + 1, 2 * i + 1), -qml.math.sqrt(1 - f))

    return unitary

def make_Q(A, R):
    r"""Calculates the :math:`\mathcal{Q}` matrix that encodes the expectation value according to
    the probability unitary :math:`\mathcal{A}` and the function-encoding unitary
    :math:`\mathcal{R}`.

    Following `this <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.022321>`__ paper,
    the expectation value is encoded as the phase of an eigenvalue of :math:`\mathcal{Q}`. This
    phase can be estimated using quantum phase estimation using the
    :func:`~.QuantumPhaseEstimation` template. See :func:`~.QuantumMonteCarlo` for more details,
    which loads ``make_Q()`` internally and applies phase estimation.

    Args:
        A (array): The unitary matrix of :math:`\mathcal{A}` which encodes the probability
            distribution
        R (array): The unitary matrix of :math:`\mathcal{R}` which encodes the function

    Returns:
        array: the :math:`\mathcal{Q}` unitary
    """
    A_big = qml.math.kron(A, np.eye(2))
    F = R @ A_big
    F_dagger = F.conj().T

    dim = len(R)
    V = _make_V(dim)
    Z = _make_Z(dim)
    UV = F @ Z @ F_dagger @ V

    return UV @ UV

