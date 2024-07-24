class _Tableau(NamedTuple):
    """To specify a particular method, one needs to provide the integer s
    (the number of stages), and the coefficients a[i,j] (for 1 ≤ j < i ≤ s),
    b[i] (for i = 1, 2, ..., s) and c[i] (for i = 2, 3, ..., s). The matrix
    [aij] is called the Runge–Kutta matrix, while the b[i] and c[i] are known
    as the weights and the nodes. These data are usually arranged in a
    mnemonic device, known as a Butcher tableau (after John C. Butcher):

    Examples
    --------
    >>> euler = _Tableau(c=[0.0],
    ...                  b=[1.0],
    ...                  a=[[0.0]]
    ... )
    >>> euler.c
    [0.0]

    Attributes
    ----------
    c: List[float]
        The nodes
    b: List[float]
        The weights
    a: List[List[float]]
        The Runge-Kutta matrix

    """
    c: List[float]
    b: List[float]
    a: List[List[float]]

