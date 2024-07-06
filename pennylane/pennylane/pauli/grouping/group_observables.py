def group_observables(observables, coefficients=None, grouping_type="qwc", method="rlf"):
    """Partitions a list of observables (Pauli operations and tensor products thereof) into
    groupings according to a binary relation (qubit-wise commuting, fully-commuting, or
    anticommuting).

    Partitions are found by 1) mapping the list of observables to a graph where vertices represent
    observables and edges encode the binary relation, then 2) solving minimum clique cover for the
    graph using graph-colouring heuristic algorithms.

    Args:
        observables (list[Observable]): a list of Pauli word ``Observable`` instances (Pauli
            operation instances and :class:`~.Tensor` instances thereof)
        coefficients (tensor_like): A tensor or list of coefficients. If not specified,
            output ``partitioned_coeffs`` is not returned.
        grouping_type (str): The type of binary relation between Pauli words.
            Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
        method (str): the graph coloring heuristic to use in solving minimum clique cover, which
            can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest First)

    Returns:
       tuple:

           * list[list[Observable]]: A list of the obtained groupings. Each grouping
             is itself a list of Pauli word ``Observable`` instances.
           * list[tensor_like]: A list of coefficient groupings. Each coefficient
             grouping is itself a tensor or list of the grouping's corresponding coefficients. This is only
             returned if coefficients are specified.

    Raises:
        IndexError: if the input list of coefficients is not of the same length as the input list
            of Pauli words

    **Example**

    >>> obs = [qml.Y(0), qml.X(0) @ qml.X(1), qml.Z(1)]
    >>> coeffs = [1.43, 4.21, 0.97]
    >>> obs_groupings, coeffs_groupings = group_observables(obs, coeffs, 'anticommuting', 'lf')
    >>> obs_groupings
    [[Z(1), X(0) @ X(1)],
     [Y(0)]]
    >>> coeffs_groupings
    [[0.97, 4.21], [1.43]]
    """

    if coefficients is not None:
        if qml.math.shape(coefficients)[0] != len(observables):
            raise IndexError(
                "The coefficients list must be the same length as the observables list."
            )

    no_wires_obs = []
    wires_obs = []
    for ob in observables:
        if len(ob.wires) == 0:
            no_wires_obs.append(ob)
        else:
            wires_obs.append(ob)
    if not wires_obs:
        if coefficients is None:
            return [no_wires_obs]
        return [no_wires_obs], [coefficients]

    pauli_grouping = PauliGroupingStrategy(
        wires_obs, grouping_type=grouping_type, graph_colourer=method
    )

    temp_opmath = not qml.operation.active_new_opmath() and any(
        isinstance(o, (Prod, SProd)) for o in observables
    )
    if temp_opmath:
        qml.operation.enable_new_opmath(warn=False)

    try:
        partitioned_paulis = pauli_grouping.colour_pauli_graph()
    finally:
        if temp_opmath:
            qml.operation.disable_new_opmath(warn=False)

    partitioned_paulis[0].extend(no_wires_obs)

    if coefficients is None:
        return partitioned_paulis

    partitioned_coeffs = _partition_coeffs(partitioned_paulis, observables, coefficients)

    return partitioned_paulis, partitioned_coeffs

