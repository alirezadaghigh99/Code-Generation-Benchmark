def QGTOnTheFly(
    vstate, *, chunk_size=None, holomorphic: Optional[bool] = None, **kwargs
) -> "QGTOnTheFlyT":
    """
    Lazy representation of an S Matrix computed by performing 2 jvp
    and 1 vjp products, using the variational state's model, the
    samples that have already been computed, and the vector.

    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contained in
    the field `sr`.

    Args:
        vstate: The variational State.
        chunk_size: If supplied, overrides the chunk size of the variational state
                    (useful for models where the backward pass requires more
                    memory than the forward pass).
    """
    if kwargs.pop("diag_scale", None) is not None:
        raise NotImplementedError(
            "\n`diag_scale` argument is not yet supported by QGTOnTheFly."
            "Please use `QGTJacobianPyTree` or `QGTJacobianDense`.\n\n"
            "You are also encouraged to nag the developers to support "
            "this feature.\n\n"
        )

    # TODO: Find a better way to handle this case
    from netket.vqs import FullSumState

    if isinstance(vstate, FullSumState):
        samples = split_array_mpi(vstate._all_states)
        pdf = split_array_mpi(vstate.probability_distribution())
    else:
        samples = vstate.samples
        pdf = None

    if chunk_size is None:
        chunk_size = getattr(vstate, "chunk_size", None)

    return QGTOnTheFly_DefaultConstructor(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        samples,
        pdf=pdf,
        chunk_size=chunk_size,
        holomorphic=holomorphic,
        **kwargs,
    )

