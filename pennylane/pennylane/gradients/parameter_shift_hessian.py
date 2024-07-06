def _process_argnum(argnum, tape):
    """Process the argnum keyword argument to ``param_shift_hessian`` from any of ``None``,
    ``int``, ``Sequence[int]``, ``array_like[bool]`` to an ``array_like[bool]``."""
    _trainability_note = (
        "This may be caused by attempting to differentiate with respect to parameters "
        "that are not marked as trainable."
    )
    if argnum is None:
        # All trainable tape parameters are considered
        argnum = list(range(tape.num_params))
    elif isinstance(argnum, int):
        if argnum >= tape.num_params:
            raise ValueError(
                f"The index {argnum} exceeds the number of trainable tape parameters "
                f"({tape.num_params}). " + _trainability_note
            )
        # Make single marked parameter an iterable
        argnum = [argnum]

    if len(qml.math.shape(argnum)) == 1:
        # If the iterable is 1D, consider all combinations of all marked parameters
        if not qml.math.array(argnum).dtype == bool:
            # If the 1D iterable contains indices, make sure it contains valid indices...
            if qml.math.max(argnum) >= tape.num_params:
                raise ValueError(
                    f"The index {qml.math.max(argnum)} exceeds the number of "
                    f"trainable tape parameters ({tape.num_params})." + _trainability_note
                )
            # ...and translate it to Boolean 1D iterable
            argnum = [i in argnum for i in range(tape.num_params)]
        elif len(argnum) != tape.num_params:
            # If the 1D iterable already is Boolean, check its length
            raise ValueError(
                "One-dimensional Boolean array argnum is expected to have as many entries as the "
                f"tape has trainable parameters ({tape.num_params}), but got {len(argnum)}."
                + _trainability_note
            )
        # Finally mark all combinations using the outer product
        argnum = qml.math.tensordot(argnum, argnum, axes=0)

    elif not (
        qml.math.shape(argnum) == (tape.num_params,) * 2
        and qml.math.array(argnum).dtype == bool
        and qml.math.allclose(qml.math.transpose(argnum), argnum)
    ):
        # If the iterable is 2D, make sure it is Boolean, symmetric and of the correct size
        raise ValueError(
            f"Expected a symmetric 2D Boolean array with shape {(tape.num_params,) * 2} "
            f"for argnum, but received {argnum}." + _trainability_note
        )
    return argnum

