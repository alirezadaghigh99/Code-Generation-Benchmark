def _decompose_multicontrolled_unitary(op, control_wires):
    """Decomposes general multi controlled unitary with no work wires
    Follows approach from Lemma 7.5 combined with 7.3 and 7.2 of
    https://arxiv.org/abs/quant-ph/9503016.

    We are assuming this decomposition is used only in the general cases
    """
    if not op.has_matrix or len(op.wires) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation with a matrix representation"
        )

    target_wire = op.wires
    if len(control_wires) == 0:
        return [op]
    if len(control_wires) == 1:
        return ctrl_decomp_zyz(op, control_wires)
    if _is_single_qubit_special_unitary(op):
        return ctrl_decomp_bisect(op, control_wires)
    # use recursive decomposition of general gate
    return _decompose_recursive(op, 1.0, control_wires, target_wire, Wires([]))

