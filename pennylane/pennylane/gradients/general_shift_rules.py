def generate_shift_rule(frequencies, shifts=None, order=1):
    r"""Computes the parameter shift rule for a unitary based on its generator's eigenvalue
    frequency spectrum.

    To compute gradients of circuit parameters in variational quantum algorithms, expressions for
    cost function first derivatives with respect to the variational parameters can be cast into
    linear combinations of expectation values at shifted parameter values. The coefficients and
    shifts defining the linear combination can be obtained from the unitary generator's eigenvalue
    frequency spectrum. Details can be found in
    `Wierichs et al. (2022) <https://doi.org/10.22331/q-2022-03-30-677>`__.

    Args:
        frequencies (tuple[int or float]): The tuple of eigenvalue frequencies. Eigenvalue
            frequencies are defined as the unique positive differences obtained from a set of
            eigenvalues.
        shifts (tuple[int or float]): the tuple of shift values. If unspecified,
            equidistant shifts are assumed. If supplied, the length of this tuple should match the
            number of given frequencies.
        order (int): the order of differentiation to compute the shift rule for

    Returns:
        tuple: a tuple of coefficients and shifts describing the gradient rule for the
        parameter-shift method. For parameter :math:`\phi`, the coefficients :math:`c_i` and the
        shifts :math:`s_i` combine to give a gradient rule of the following form:

        .. math:: \frac{\partial}{\partial\phi}f = \sum_{i} c_i f(\phi + s_i).

        where :math:`f(\phi) = \langle 0|U(\phi)^\dagger \hat{O} U(\phi)|0\rangle`
        for some observable :math:`\hat{O}` and the unitary :math:`U(\phi)=e^{iH\phi}`.

    Raises:
        ValueError: if ``frequencies`` is not a list of unique positive values, or if ``shifts``
            (if specified) is not a list of unique values the same length as ``frequencies``.

    **Examples**

    An example of obtaining the frequencies from generator eigenvalues, and obtaining the parameter
    shift rule:

    >>> eigvals = (-0.5, 0, 0, 0.5)
    >>> frequencies = eigvals_to_frequencies(eigvals)
    >>> generate_shift_rule(frequencies)
    array([[ 0.4267767 ,  1.57079633],
           [-0.4267767 , -1.57079633],
           [-0.0732233 ,  4.71238898],
           [ 0.0732233 , -4.71238898]])

    An example with explicitly specified shift values:

    >>> frequencies = (1, 2, 4)
    >>> shifts = (np.pi / 3, 2 * np.pi / 3, np.pi / 4)
    >>> generate_shift_rule(frequencies, shifts)
    array([[ 3.        ,  0.78539816],
           [-3.        , -0.78539816],
           [-2.09077028,  1.04719755],
           [ 2.09077028, -1.04719755],
           [ 0.2186308 ,  2.0943951 ],
           [-0.2186308 , -2.0943951 ]])

    Higher order shift rules (corresponding to the :math:`n`-th derivative of the parameter) can be
    requested via the ``order`` argument. For example, to extract the second order shift rule for a
    gate with generator :math:`X/2`:

    >>> eigvals = (0.5, -0.5)
    >>> frequencies = eigvals_to_frequencies(eigvals)
    >>> generate_shift_rule(frequencies, order=2)
    array([[-0.5       ,  0.        ],
           [ 0.5       , -3.14159265]])

    This corresponds to the shift rule
    :math:`\frac{\partial^2 f}{\partial \phi^2} = \frac{1}{2} \left[f(\phi) - f(\phi-\pi)\right]`.
    """
    frequencies = tuple(f for f in frequencies if f > 0)
    rule = _get_shift_rule(frequencies, shifts=shifts)

    if order > 1:
        T = frequencies_to_period(frequencies)
        rule = _iterate_shift_rule(rule, order, period=T)

    return process_shifts(rule, tol=1e-10)

def frequencies_to_period(frequencies, decimals=5):
    r"""Returns the period of a Fourier series as defined
    by a set of frequencies.

    The period is simply :math:`2\pi/gcd(frequencies)`,
    where :math:`\text{gcd}` is the greatest common divisor.

    Args:
        spectra (tuple[int, float]): frequency spectra
        decimals (int): Number of decimal places to round to
            if there are non-integral frequencies.

    Returns:
        tuple[int, float]: frequencies

    **Example**

    >>> frequencies = (0.5, 1.0)
    >>> frequencies_to_period(frequencies)
    12.566370614359172
    """
    try:
        gcd = np.gcd.reduce(frequencies)

    except TypeError:
        # np.gcd only support integer frequencies
        exponent = 10**decimals
        frequencies = np.round(frequencies, decimals) * exponent
        gcd = np.gcd.reduce(np.int64(frequencies)) / exponent

    return 2 * np.pi / gcd

def generate_multi_shift_rule(frequencies, shifts=None, orders=None):
    r"""Computes the parameter shift rule with respect to two parametrized unitaries,
    given their generator's eigenvalue frequency spectrum. This corresponds to a
    shift rule that computes off-diagonal elements of higher order derivative tensors.
    For the second order, this corresponds to the Hessian.

    Args:
        frequencies (list[tuple[int or float]]): List of eigenvalue frequencies corresponding
            to the each parametrized unitary.
        shifts (list[tuple[int or float]]): List of shift values corresponding to each parametrized
            unitary. If unspecified, equidistant shifts are assumed. If supplied, the length
            of each tuple in the list must be the same as the length of the corresponding tuple in
            ``frequencies``.
        orders (list[int]): the order of differentiation for each parametrized unitary.
            If unspecified, the first order derivative shift rule is computed for each parametrized
            unitary.

    Returns:
        tuple: a tuple of coefficients, shifts for the first parameter, and shifts for the
        second parameter, describing the gradient rule
        for the parameter-shift method.

        For parameters :math:`\phi_a` and :math:`\phi_b`, the
        coefficients :math:`c_i` and the shifts :math:`s^{(a)}_i`, :math:`s^{(b)}_i`,
        combine to give a gradient rule of the following form:

        .. math::

            \frac{\partial^2}{\partial\phi_a \partial\phi_b}f
            = \sum_{i} c_i f(\phi_a + s^{(a)}_i, \phi_b + s^{(b)}_i).

        where :math:`f(\phi_a, \phi_b) = \langle 0|U(\phi_a)^\dagger V(\phi_b)^\dagger \hat{O} V(\phi_b) U(\phi_a)|0\rangle`
        for some observable :math:`\hat{O}` and unitaries :math:`U(\phi_a)=e^{iH_a\phi_a}` and :math:`V(\phi_b)=e^{iH_b\phi_b}`.

    **Example**

    >>> generate_multi_shift_rule([(1,), (1,)])
    array([[ 0.25      ,  1.57079633,  1.57079633],
           [-0.25      ,  1.57079633, -1.57079633],
           [-0.25      , -1.57079633,  1.57079633],
           [ 0.25      , -1.57079633, -1.57079633]])

    This corresponds to the gradient rule

    .. math::

        \begin{align*}
        \frac{\partial^2 f}{\partial x\partial y} &= \frac{1}{4}
        [f(x+\pi/2, y+\pi/2) - f(x+\pi/2, y-\pi/2)\\
        &\phantom{\frac{1}{4}[}-f(x-\pi/2, y+\pi/2) + f(x-\pi/2, y-\pi/2) ].
        \end{align*}

    """
    rules = []
    shifts = shifts or [None] * len(frequencies)
    orders = orders or [1] * len(frequencies)

    for f, s, o in zip(frequencies, shifts, orders):
        rule = generate_shift_rule(f, shifts=s, order=o)
        rules.append(process_shifts(rule))

    return _combine_shift_rules(rules)

def eigvals_to_frequencies(eigvals):
    r"""Convert an eigenvalue spectrum to frequency values, defined
    as the the set of positive, unique differences of the eigenvalues in the spectrum.

    Args:
        eigvals (tuple[int, float]): eigenvalue spectra

    Returns:
        tuple[int, float]: frequencies

    **Example**

    >>> eigvals = (-0.5, 0, 0, 0.5)
    >>> eigvals_to_frequencies(eigvals)
    (0.5, 1.0)
    """
    unique_eigvals = sorted(set(eigvals))
    return tuple({j - i for i, j in itertools.combinations(unique_eigvals, 2)})

def _iterate_shift_rule_with_multipliers(rule, order, period):
    r"""Helper method to repeat a shift rule that includes multipliers multiple
    times along the same parameter axis for higher-order derivatives."""
    combined_rules = []

    for partial_rules in itertools.product(rule, repeat=order):
        c, m, s = np.stack(partial_rules).T
        cumul_shift = 0.0
        for _m, _s in zip(m, s):
            cumul_shift *= _m
            cumul_shift += _s
        if period is not None:
            cumul_shift = np.mod(cumul_shift + 0.5 * period, period) - 0.5 * period
        combined_rules.append(np.stack([np.prod(c), np.prod(m), cumul_shift]))

    # combine all terms in the linear combination into a single
    # array, with column order (coefficients, multipliers, shifts)
    return qml.math.stack(combined_rules)

