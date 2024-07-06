def _nonzero_coeffs_and_words(coefficients, num_wires, atol=1e-8):
    """Given a tuple of coefficient tensors with a common first axis size :math:`4^N-1`,
    where :math:`N` is the number of wires, filter out those indices for the first axis for which
    the sliced coefficients are not all zero. Return these coefficients, as well as the
    corresponding Pauli word strings on :math:`N` wires.

    Args:
        coefficients(tuple[tensor_like]): Coefficients to filter.
        num_wires (int): Number of qubits the generators act on.
        atol (float): absolute tolerance used to determine whether a tensor is zero.
    """
    words = pauli_basis_strings(num_wires)
    new_coefficients = [[] for _ in coefficients]
    new_words = []
    for word, *coeffs in zip(words, *coefficients):
        if all(qml.math.allclose(c, 0.0, atol=atol, rtol=0.0) for c in coeffs):
            continue
        new_words.append(word)
        for new_coeffs, c in zip(new_coefficients, coeffs):
            new_coeffs.append(c)
    return new_coefficients, new_words

def _generate_tapes_and_coeffs(tape, idx, atol, cache):
    """Compute the modified tapes and coefficients required to compute the pulse generator
    derivative of a tape with respect to an indicated trainable parameter.

    Args:
        tape (`~.QuantumTape`): Tape to differentiate.
        idx (int): Index (referring to ``tape.trainable_parameters``) of the parameter
            with respect to which to differentiate.
        atol (float): absolute tolerance used to determine whether a coefficient is zero.
        cache (dict): Caching dictionary that allows to skip adding duplicate modified tapes.

    Returns:
        list[`~.QuantumScript`]: Modified tapes to be added to the pulse generator differentiation
            tapes. It is an empty list if modified tapes were already created for another
            parameter of the pulse of interest.
        tuple[int, int, tensor_like]: Gradient computation data, consisting of the start and end
            indices into the total list of tapes as well as the coefficients that need to be
            contracted with the corresponding results to obtain the partial derivative with respect
            to the indicated trainable parameter.
        dict: Input ``cache`` dictionary. If the cache lookup missed, the cache is extended by one
            entry and its entry ``"total_num_tapes"`` is increased by the number of created tapes.
    """
    # ``op`` is the operation into which the trainable tape parameter with index ``idx`` feeds.
    # ``op_idx`` is the index of ``op`` within all tape operations. ``term_idx`` is the index
    # of the trainable tape parameter within ``op``.
    op, op_idx, term_idx = tape.get_operation(idx)
    if op_idx in cache:
        # operation was analyzed and tapes were added before. Get tape indices and coefficients.
        start, end, all_coeffs = cache[op_idx]
        return [], (start, end, all_coeffs[term_idx]), cache

    if not isinstance(op, ParametrizedEvolution):
        # only ParametrizedEvolution can be treated with this gradient transform
        raise ValueError(
            "pulse_odegen does not support differentiating parameters of "
            f"other operations than pulses, but received operation {op}."
        )

    num_wires = len(op.wires)
    # Compute the effective generators of the given pulse _for all parameters_.
    generators = _one_parameter_generators(op)
    # Obtain one-parameter-group coefficients in Pauli basis and filter for non-zero coeffs
    all_coeffs = _one_parameter_paulirot_coeffs(generators, num_wires)
    all_coeffs, pauli_words = _nonzero_coeffs_and_words(all_coeffs, num_wires, atol)
    # create PauliRot gates for each Pauli word (with a non-zero coefficient) and for both shifts
    pauli_rots = [
        qml.PauliRot(angle, word, wires=op.wires)
        for word in pauli_words
        for angle in [np.pi / 2, -np.pi / 2]
    ]
    # create tapes with the above PauliRot gates inserted, one per tape
    tapes = _insert_op(tape, pauli_rots, op_idx)
    # get the previous total number of tapes from the cache and determine start and end indices
    end = (start := cache["total_num_tapes"]) + (num_tapes := len(tapes))
    # store the tape indices and coefficients for this operation, to be retrieved from the cache
    # for other parameters of the same operation, see first if clause above.
    cache[op_idx] = (start, end, all_coeffs)
    # increment total number of tapes
    cache["total_num_tapes"] += num_tapes
    # Return the start and end pointer for the created tapes/the corresponding results.
    # Also return the coefficients for the current trainable parameter of the pulse
    return tapes, (start, end, all_coeffs[term_idx]), cache

