def compute_vjp_multi(dy, jac, num=None):
    """Convenience function to compute the vector-Jacobian product for a given
    vector of gradient outputs and a Jacobian for a tape with multiple measurements.

    Args:
        dy (tensor_like): vector of gradient outputs
        jac (tensor_like, tuple): Jacobian matrix
        num (int): The length of the flattened ``dy`` argument. This is an
            optional argument, but can be useful to provide if ``dy`` potentially
            has no shape (for example, due to tracing or just-in-time compilation).

    Returns:
        tensor_like: the vector-Jacobian product

    **Examples**

    1. For a single parameter and multiple measurement (one without shape and one with shape, e.g. expval and probs):

    .. code-block:: pycon

        >>> jac = tuple([np.array(0.1), np.array([0.3, 0.4])])
        >>> dy = tuple([np.array(1.0), np.array([1.0, 2.0])])
        >>> compute_vjp_multi(dy, jac)
        array([1.2])

    2. For multiple parameters (in this case 2 parameters) and multiple measurement (one without shape and one with
    shape, e.g. expval and probs):

    .. code-block:: pycon

        >>> jac = ((np.array(0.1), np.array(0.2)), (np.array([0.3, 0.4]), np.array([0.5, 0.6])))
        >>> dy = tuple([np.array(1.0), np.array([1.0, 2.0])])
        >>> compute_vjp_multi(dy, jac)
        array([1.2, 1.9])

    """
    if jac is None:
        return None

    # Single parameter
    if not isinstance(jac[0], (tuple, autograd.builtins.SequenceBox)):
        res = []
        for d, j_ in zip(dy, jac):
            res.append(compute_vjp_single(d, j_, num=num))
        res = qml.math.sum(qml.math.stack(res), axis=0)
    # Multiple parameters
    else:
        try:
            dy_interface = qml.math.get_interface(dy[0])
            # dy  -> (i,j)      observables, entries per observable
            # jac -> (i,k,j)    observables, parameters, entries per observable
            # Contractions over observables and entries per observable
            dy_shape = qml.math.shape(dy)
            if len(dy_shape) > 1:  # multiple values exist per observable output
                return qml.math.array(qml.math.einsum("ij,i...j", dy, jac), like=dy[0])

            if dy_interface == "tensorflow":
                # TF needs a different path for Hessian support
                return qml.math.array(
                    qml.math.einsum("i,i...", dy, jac, like=dy[0]), like=dy[0]
                )  # Scalar value per observable output
            return qml.math.array(
                qml.math.einsum("i,i...", dy, jac), like=dy[0]
            )  # Scalar value per observable output
        # NOTE: We want any potential failure to fall back here, so catch every exception type
        # TODO: Catalogue and update for expected exception types
        except Exception:  # pylint: disable=broad-except
            res = []
            for d, j_ in zip(dy, jac):
                sub_res = []
                for j in j_:
                    sub_res.append(qml.math.squeeze(compute_vjp_single(d, j, num=num)))
                res.append(sub_res)
            res = qml.math.stack([qml.math.stack(r) for r in res])
            res = qml.math.sum(res, axis=0)
    return res

