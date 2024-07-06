def bind_new_parameters(op: Operator, params: Sequence[TensorLike]) -> Operator:
    """Create a new operator with updated parameters

    This function takes an :class:`~.Operator` and new parameters as input and
    returns a new operator of the same type with the new parameters. This function
    does not mutate the original operator.

    Args:
        op (.Operator): Operator to update
        params (Sequence[TensorLike]): New parameters to create operator with. This
            must have the same shape as `op.data`.

    Returns:
        .Operator: New operator with updated parameters
    """
    try:
        return op.__class__(*params, wires=op.wires, **copy.deepcopy(op.hyperparameters))
    except (TypeError, ValueError):
        # operation is doing something different with its call signature.
        new_op = copy.deepcopy(op)
        new_op.data = tuple(params)
        return new_op

