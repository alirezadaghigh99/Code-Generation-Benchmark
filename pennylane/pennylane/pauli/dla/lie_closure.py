def lie_closure(
    generators: Iterable[Union[PauliWord, PauliSentence, Operator]],
    max_iterations: int = 10000,
    verbose: bool = False,
    pauli: bool = False,
    tol: float = None,
) -> Iterable[Union[PauliWord, PauliSentence, Operator]]:
    r"""Compute the dynamical Lie algebra from a set of generators.

    The Lie closure, pronounced "Lee" closure, is a way to compute the so-called dynamical Lie algebra (DLA) of a set of generators :math:`\mathcal{G} = \{G_1, .. , G_N\}`.
    For such generators, one computes all nested commutators :math:`[G_i, [G_j, .., [G_k, G_\ell]]]` until no new operators are generated from commutation.
    All these operators together form the DLA, see e.g. section IIB of `arXiv:2308.01432 <https://arxiv.org/abs/2308.01432>`__.

    Args:
        generators (Iterable[Union[PauliWord, PauliSentence, Operator]]): generating set for which to compute the
            Lie closure.
        max_iterations (int): maximum depth of nested commutators to consider. Default is ``10000``.
        verbose (bool): whether to print out progress updates during Lie closure
            calculation. Default is ``False``.
        pauli (bool): Indicates whether it is assumed that :class:`~.PauliSentence` or :class:`~.PauliWord` instances are input and returned.
            This can help with performance to avoid unnecessary conversions to :class:`~pennylane.operation.Operator`
            and vice versa. Default is ``False``.
        tol (float): Numerical tolerance for the linear independence check used in :class:`~.PauliVSpace`.

    Returns:
        Union[list[:class:`~.PauliSentence`], list[:class:`~.Operator`]]: a basis of either :class:`~.PauliSentence` or :class:`~.Operator` instances that is closed under
        commutators (Lie closure).

    .. seealso:: :func:`~structure_constants`, :func:`~center`, :class:`~pennylane.pauli.PauliVSpace`, `Demo: Introduction to Dynamical Lie Algebras for quantum practitioners <https://pennylane.ai/qml/demos/tutorial_liealgebra/>`__

    **Example**

    Let us walk through a simple example of computing the Lie closure of the generators of the transverse field Ising model on two qubits.

    >>> ops = [X(0) @ X(1), Z(0), Z(1)]

    A first round of commutators between all elements yields:

    >>> qml.commutator(X(0) @ X(1), Z(0))
    -2j * (Y(0) @ X(1))
    >>> qml.commutator(X(0) @ X(1), Z(1))
    -2j * (X(0) @ Y(1))

    A next round of commutators between all elements further yields the new operator ``Y(0) @ Y(1)``.

    >>> qml.commutator(X(0) @ Y(1), Z(0))
    -2j * (Y(0) @ Y(1))

    After that, no new operators emerge from taking nested commutators and we have the resulting DLA.
    This can be done in short via ``lie_closure`` as follows.

    >>> ops = [X(0) @ X(1), Z(0), Z(1)]
    >>> dla = qml.lie_closure(ops)
    >>> print(dla)
    [X(1) @ X(0),
     Z(0),
     Z(1),
     -1.0 * (Y(0) @ X(1)),
     -1.0 * (X(0) @ Y(1)),
     -1.0 * (Y(0) @ Y(1))]

    Note that we normalize by removing the factors of :math:`2i`, though minus signs are left intact.

    .. details::
        :title: Usage Details

        Note that by default, ``lie_closure`` returns PennyLane operators. Internally we use the more
        efficient representation in terms of :class:`~pennylane.pauli.PauliSentence` by making use of the ``op.pauli_rep``
        attribute of operators composed of Pauli operators. If desired, this format can be returned by using
        the keyword ``pauli=True``. In that case, the input is also assumed to be a :class:`~pennylane.pauli.PauliSentence` instance.

        >>> ops = [
        ...     PauliSentence({PauliWord({0: "X", 1: "X"}): 1.}),
        ...     PauliSentence({PauliWord({0: "Z"}): 1.}),
        ...     PauliSentence({PauliWord({1: "Z"}): 1.}),
        ... ]
        >>> dla = qml.lie_closure(ops, pauli=True)
        >>> print(dla)
        [1.0 * X(0) @ X(1),
         1.0 * Z(0),
         1.0 * Z(1),
         -1.0 * Y(0) @ X(1),
         -1.0 * X(0) @ Y(1),
         -1.0 * Y(0) @ Y(1)]
        >>> type(dla[0])
        pennylane.pauli.pauli_arithmetic.PauliSentence

    """
    if not all(isinstance(op, (PauliSentence, PauliWord)) for op in generators):
        if pauli:
            raise TypeError(
                "All generators need to be of type PauliSentence or PauliWord when using pauli=True in lie_closure."
            )

        generators = [
            rep if (rep := op.pauli_rep) is not None else qml.pauli.pauli_sentence(op)
            for op in generators
        ]

    vspace = PauliVSpace(generators, tol=tol)

    epoch = 0
    old_length = 0  # dummy value
    new_length = len(vspace)

    while (new_length > old_length) and (epoch < max_iterations):
        if verbose:
            print(f"epoch {epoch+1} of lie_closure, DLA size is {new_length}")
        for ps1, ps2 in itertools.combinations(vspace.basis, 2):
            com = ps1.commutator(ps2)
            if len(com) == 0:  # skip because operators commute
                continue

            # result is always purely imaginary
            # remove common factor 2 with Pauli commutators
            for pw, val in com.items():
                com[pw] = val.imag / 2
            vspace.add(com, tol=tol)

        # Updated number of linearly independent PauliSentences from previous and current step
        old_length = new_length
        new_length = len(vspace)
        epoch += 1

    if verbose > 0:
        print(f"After {epoch} epochs, reached a DLA size of {new_length}")

    res = vspace.basis
    if not pauli:
        res = [op.operation() for op in res]

    return res

