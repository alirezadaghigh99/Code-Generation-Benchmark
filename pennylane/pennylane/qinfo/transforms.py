def _compute_cfim(p, dp):
    r"""Computes the (num_params, num_params) classical fisher information matrix from the probabilities and its derivatives
    I.e. it computes :math:`classical_fisher_{ij} = \sum_\ell (\partial_i p_\ell) (\partial_i p_\ell) / p_\ell`
    """
    # Exclude values where p=0 and calculate 1/p
    nonzeros_p = qml.math.where(p > 0, p, qml.math.ones_like(p))
    one_over_p = qml.math.where(p > 0, qml.math.ones_like(p), qml.math.zeros_like(p))
    one_over_p = one_over_p / nonzeros_p

    # Multiply dp and p
    # Note that casting and being careful about dtypes is necessary as interfaces
    # typically treat derivatives (dp) with float32, while standard execution (p) comes in float64
    dp = qml.math.cast_like(dp, p)
    dp = qml.math.reshape(
        dp, (len(p), -1)
    )  # Squeeze does not work, as you could have shape (num_probs, num_params) with num_params = 1
    dp_over_p = qml.math.transpose(dp) * one_over_p  # creates (n_params, n_probs) array

    # (n_params, n_probs) @ (n_probs, n_params) = (n_params, n_params)
    return dp_over_p @ dpdef trace_distance(qnode0, qnode1, wires0, wires1):
    r"""
    Compute the trace distance for two :class:`.QNode` returning a :func:`~pennylane.state` (a state can be a state vector
    or a density matrix, depending on the device) acting on quantum systems with the same size.

    .. math::
        T(\rho, \sigma)=\frac12\|\rho-\sigma\|_1
        =\frac12\text{Tr}\left(\sqrt{(\rho-\sigma)^{\dagger}(\rho-\sigma)}\right)

    where :math:`\|\cdot\|_1` is the Schatten :math:`1`-norm.

    The trace distance measures how close two quantum states are. In particular, it upper-bounds
    the probability of distinguishing two quantum states.

    Args:
        qnode0 (QNode): A :class:`.QNode` returning a :func:`~pennylane.state`.
        qnode1 (QNode): A :class:`.QNode` returning a :func:`~pennylane.state`.
        wires0 (Sequence[int]): the subsystem of the first QNode.
        wires1 (Sequence[int]): the subsystem of the second QNode.

    Returns:
        func: A function that takes as input the joint arguments of the two QNodes,
        and returns the trace distance between their output states.

    **Example**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

    The ``qml.qinfo.trace_distance`` transform can be used to compute the trace distance
    between the output states of the QNode:

    >>> trace_distance_circuit = qml.qinfo.trace_distance(circuit, circuit, wires0=[0], wires1=[0])

    The returned function takes two tuples as input, the first being the arguments to the
    first QNode and the second being the arguments to the second QNode:

    >>> x, y = np.array(0.4), np.array(0.6)
    >>> trace_distance_circuit((x,), (y,))
    0.047862689546603415

    This transform is fully differentiable:

    .. code-block:: python

        def wrapper(x, y):
            return trace_distance_circuit((x,), (y,))

    >>> wrapper(x, y)
    0.047862689546603415
    >>> qml.grad(wrapper)(x, y)
    (tensor(-0.19470917, requires_grad=True),
     tensor(0.28232124, requires_grad=True))
    """

    if len(wires0) != len(wires1):
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    state_qnode0 = qml.qinfo.reduced_dm(qnode0, wires=wires0)
    state_qnode1 = qml.qinfo.reduced_dm(qnode1, wires=wires1)

    def evaluate_trace_distance(all_args0=None, all_args1=None):
        """Wrapper used for evaluation of the trace distance between two states computed from
        QNodes. It allows giving the args and kwargs to each :class:`.QNode`.

        Args:
            all_args0 (tuple): Tuple containing the arguments (*args, kwargs) of the first :class:`.QNode`.
            all_args1 (tuple): Tuple containing the arguments (*args, kwargs) of the second :class:`.QNode`.

        Returns:
            float: Trace distance between two quantum states
        """
        if not isinstance(all_args0, tuple) and all_args0 is not None:
            all_args0 = (all_args0,)

        if not isinstance(all_args1, tuple) and all_args1 is not None:
            all_args1 = (all_args1,)

        # If no all_args is given, evaluate the QNode without args
        if all_args0 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args0[-1], dict):
                args0 = all_args0[:-1]
                kwargs0 = all_args0[-1]
            else:
                args0 = all_args0
                kwargs0 = {}
            state0 = state_qnode0(*args0, **kwargs0)
        else:
            # No args
            state0 = state_qnode0()

        # If no all_args is given, evaluate the QNode without args
        if all_args1 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args1[-1], dict):
                args1 = all_args1[:-1]
                kwargs1 = all_args1[-1]
            else:
                args1 = all_args1
                kwargs1 = {}
            state1 = state_qnode1(*args1, **kwargs1)
        else:
            # No args
            state1 = state_qnode1()

        # From the two generated states, compute the trace distance
        return qml.math.trace_distance(state0, state1)

    return evaluate_trace_distance