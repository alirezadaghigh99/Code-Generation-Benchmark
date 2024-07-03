def _parshift_and_integrate(
    results,
    cjacs,
    int_prefactor,
    psr_coeffs,
    single_measure,
    has_partitioned_shots,
    use_broadcasting,
):
    """Apply the parameter-shift rule post-processing to tape results and contract
    with classical Jacobians, effectively evaluating the numerical integral of the stochastic
    parameter-shift rule.

    Args:
        results (list): Tape evaluation results, corresponding to the modified quantum
            circuit result when using the applicable parameter shifts and the sample splitting
            times. Results should be ordered such that the different shifted circuits for a given
            splitting time are grouped together
        cjacs (tensor_like): classical Jacobian evaluated at the splitting times
        int_prefactor (float): prefactor of the numerical integration, corresponding to the size
            of the time range divided by the number of splitting time samples
        psr_coeffs (tensor_like or tuple[tensor_like]): Coefficients of the parameter-shift
            rule to contract the results with before integrating numerically.
        single_measure (bool): Whether the results contain a single measurement per shot setting
        has_partitioned_shots (bool): Whether the results have a shot vector axis
        use_broadcasting (bool): Whether broadcasting was used in the tapes that returned the
            ``results``.
    Returns:
        tensor_like or tuple[tensor_like] or tuple[tuple[tensor_like]]: Gradient entry
    """

    def _contract(coeffs, res, cjac):
        """Contract three tensors, the first two like a standard matrix multiplication
        and the result with the third tensor along the first axes."""
        return jnp.tensordot(jnp.tensordot(coeffs, res, axes=1), cjac, axes=[[0], [0]])

    if isinstance(psr_coeffs, tuple):
        num_shifts = [len(c) for c in psr_coeffs]

        def _psr_and_contract(res_list, cjacs, int_prefactor):
            """Execute the parameter-shift rule and contract with classical Jacobians.
            This function assumes multiple generating terms for the pulse parameter
            of interest"""
            res = jnp.stack(res_list)
            idx = 0

            # Preprocess the results: Reshape, create slices for different generating terms
            if use_broadcasting:
                # Slice the results according to the different generating terms. Slice away the
                # first and last value for each term, which correspond to the initial condition
                # and the final value of the time evolution, but not to splitting times
                res = tuple(res[idx : (idx := idx + n), 1:-1] for n in num_shifts)
            else:
                shape = jnp.shape(res)
                num_taus = shape[0] // sum(num_shifts)
                # Reshape the slices of the results corresponding to different generating terms.
                # Afterwards the first axis corresponds to the splitting times and the second axis
                # corresponds to the different shifts of the respective term.
                # Finally move the shifts-axis to the first position of each term.
                res = tuple(
                    jnp.moveaxis(
                        jnp.reshape(
                            res[idx : (idx := idx + n * num_taus)], (num_taus, n) + shape[1:]
                        ),
                        1,
                        0,
                    )
                    for n in num_shifts
                )

            # Contract the results, parameter-shift rule coefficients and (classical) Jacobians,
            # and include the rescaling factor from the Monte Carlo integral and from global
            # prefactors of Pauli word generators.
            diff_per_term = jnp.array(
                [_contract(c, r, cjac) for c, r, cjac in zip(psr_coeffs, res, cjacs)]
            )
            return qml.math.sum(diff_per_term, axis=0) * int_prefactor

    else:
        num_shifts = len(psr_coeffs)

        def _psr_and_contract(res_list, cjacs, int_prefactor):
            """Execute the parameter-shift rule and contract with classical Jacobians.
            This function assumes a single generating term for the pulse parameter
            of interest"""
            res = jnp.stack(res_list)

            # Preprocess the results: Reshape, create slices for different generating terms
            if use_broadcasting:
                # Slice away the first and last values, corresponding to the initial condition
                # and the final value of the time evolution, but not to splitting times
                res = res[:, 1:-1]
            else:
                # Reshape the results such that the first axis corresponds to the splitting times
                # and the second axis corresponds to different shifts. All other axes are untouched.
                # Afterwards move the shifts-axis to the first position.
                shape = jnp.shape(res)
                new_shape = (shape[0] // num_shifts, num_shifts) + shape[1:]
                res = jnp.moveaxis(jnp.reshape(res, new_shape), 1, 0)

            # Contract the results, parameter-shift rule coefficients and (classical) Jacobians,
            # and include the rescaling factor from the Monte Carlo integral and from global
            # prefactors of Pauli word generators.
            return _contract(psr_coeffs, res, cjacs) * int_prefactor

    nesting_layers = (not single_measure) + has_partitioned_shots
    if nesting_layers == 1:
        return tuple(_psr_and_contract(r, cjacs, int_prefactor) for r in zip(*results))
    if nesting_layers == 0:
        # Single measurement without shot vector
        return _psr_and_contract(results, cjacs, int_prefactor)

    # Multiple measurements with shot vector. Not supported with broadcasting yet.
    if use_broadcasting:
        # TODO: Remove once #2690 is resolved
        raise NotImplementedError(
            "Broadcasting, multiple measurements and shot vectors are currently not "
            "supported all simultaneously by stoch_pulse_grad."
        )
    return tuple(
        tuple(_psr_and_contract(_r, cjacs, int_prefactor) for _r in zip(*r)) for r in zip(*results)
    )