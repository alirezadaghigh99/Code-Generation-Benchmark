class QutritChannel(Channel):
    r"""
    Apply an arbitrary fixed qutrit channel.

    Kraus matrices that represent the fixed channel are provided
    as a list of NumPy arrays.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        K_list (list[array[complex]]): list of Kraus matrices
        wires (Union[Wires, Sequence[int], or int]): the wire(s) the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, K_list, wires=None, id=None):
        super().__init__(*K_list, wires=wires, id=id)

        # check all Kraus matrices are square matrices
        if any(K.shape[0] != K.shape[1] for K in K_list):
            raise ValueError(
                "Only channels with the same input and output Hilbert space dimensions can be applied."
            )

        # check all Kraus matrices have the same shape
        if any(K.shape != K_list[0].shape for K in K_list):
            raise ValueError("All Kraus matrices must have the same shape.")

        # check the dimension of all Kraus matrices are valid
        kraus_dim = QUDIT_DIM ** len(self.wires)
        if any(K.shape[0] != kraus_dim for K in K_list):
            raise ValueError(f"Shape of all Kraus matrices must be ({kraus_dim},{kraus_dim}).")

        # check that the channel represents a trace-preserving map
        if not any(math.is_abstract(K) for K in K_list):
            K_arr = math.array(K_list)
            Kraus_sum = math.einsum("ajk,ajl->kl", K_arr.conj(), K_arr)
            if not math.allclose(Kraus_sum, math.eye(K_list[0].shape[0])):
                raise ValueError("Only trace preserving channels can be applied.")

    def _flatten(self):
        return (self.data,), (self.wires, ())

    @staticmethod
    def compute_kraus_matrices(*kraus_matrices):  # pylint:disable=arguments-differ
        """Kraus matrices representing the QutritChannel channel.

        Args:
            *K_list (list[array[complex]]): list of Kraus matrices

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> K_list = qml.QutritDepolarizingChannel(0.75, wires=0).kraus_matrices()
        >>> res = qml.QutritChannel.compute_kraus_matrices(K_list)
        >>> all(np.allclose(r, k) for r, k  in zip(res, K_list))
        True
        """
        return list(kraus_matrices)class QutritAmplitudeDamping(Channel):
    r"""
    Single-qutrit amplitude damping error channel.

    Interaction with the environment can lead to changes in the state populations of a qutrit.
    This can be modelled by the qutrit amplitude damping channel with the following Kraus matrices:

    .. math::
        K_0 = \begin{bmatrix}
                1 & 0 & 0\\
                0 & \sqrt{1-\gamma_{10}} & 0 \\
                0 & 0 & \sqrt{1-(\gamma_{20}+\gamma_{21})}
                \end{bmatrix}

    .. math::
        K_1 = \begin{bmatrix}
                0 & \sqrt{\gamma_{10}} & 0 \\
                0 & 0 & 0 \\
                0 & 0 & 0
                \end{bmatrix}, \quad
        K_2 = \begin{bmatrix}
                0 & 0 & \sqrt{\gamma_{20}} \\
                0 & 0 & 0 \\
                0 & 0 & 0
                \end{bmatrix}, \quad
        K_3 = \begin{bmatrix}
                0 & 0 & 0 \\
                0 & 0 & \sqrt{\gamma_{21}} \\
                0 & 0 & 0
                \end{bmatrix}

    where :math:`\gamma_{10}, \gamma_{20}, \gamma_{21} \in [0, 1]` are the amplitude damping
    probabilities for subspaces (0,1), (0,2), and (1,2) respectively.

    .. note::

        When :math:`\gamma_{21}=0` then Kraus operators :math:`\{K_0, K_1, K_2\}` are adapted from
        [`1 <https://doi.org/10.48550/arXiv.1905.10481>`_] (Eq. 8).

        The Kraus operator :math:`K_3` represents the :math:`|2 \rangle \rightarrow |1 \rangle` transition which is more
        likely on some devices [`2 <https://arxiv.org/abs/2003.03307>`_] (Sec II.A).

        To maintain normalization :math:`\gamma_{20} + \gamma_{21} \leq 1`.


    **Details:**

    * Number of wires: 1
    * Number of parameters: 3

    Args:
        gamma_10 (float): :math:`|1 \rangle \rightarrow |0 \rangle` amplitude damping probability.
        gamma_20 (float): :math:`|2 \rangle \rightarrow |0 \rangle` amplitude damping probability.
        gamma_21 (float): :math:`|2 \rangle \rightarrow |1 \rangle` amplitude damping probability.
        wires (Sequence[int] or int): the wire the channel acts on.
        id (str or None): String representing the operation (optional).
    """

    num_params = 3
    num_wires = 1
    grad_method = "F"

    def __init__(self, gamma_10, gamma_20, gamma_21, wires, id=None):
        # Verify input
        for gamma in (gamma_10, gamma_20, gamma_21):
            if not math.is_abstract(gamma):
                if not 0.0 <= gamma <= 1.0:
                    raise ValueError("Each probability must be in the interval [0,1]")
        if not (math.is_abstract(gamma_20) or math.is_abstract(gamma_21)):
            if not 0.0 <= gamma_20 + gamma_21 <= 1.0:
                raise ValueError(r"\gamma_{20}+\gamma_{21} must be in the interval [0,1]")
        super().__init__(gamma_10, gamma_20, gamma_21, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(gamma_10, gamma_20, gamma_21):  # pylint:disable=arguments-differ
        r"""Kraus matrices representing the ``QutritAmplitudeDamping`` channel.

        Args:
            gamma_10 (float): :math:`|1\rangle \rightarrow |0\rangle` amplitude damping probability.
            gamma_20 (float): :math:`|2\rangle \rightarrow |0\rangle` amplitude damping probability.
            gamma_21 (float): :math:`|2\rangle \rightarrow |1\rangle` amplitude damping probability.

        Returns:
            list(array): list of Kraus matrices

        **Example**

        >>> qml.QutritAmplitudeDamping.compute_kraus_matrices(0.5, 0.25, 0.36)
        [
        array([ [1.        , 0.        , 0.        ],
                [0.        , 0.70710678, 0.        ],
                [0.        , 0.        , 0.6244998 ]]),
        array([ [0.        , 0.70710678, 0.        ],
                [0.        , 0.        , 0.        ],
                [0.        , 0.        , 0.        ]]),
        array([ [0.        , 0.        , 0.5       ],
                [0.        , 0.        , 0.        ],
                [0.        , 0.        , 0.        ]])
        array([ [0.        , 0.        , 0.        ],
                [0.        , 0.        , 0.6       ],
                [0.        , 0.        , 0.        ]])
        ]
        """
        K0 = math.diag(
            [1, math.sqrt(1 - gamma_10 + math.eps), math.sqrt(1 - gamma_20 - gamma_21 + math.eps)]
        )
        K1 = math.sqrt(gamma_10 + math.eps) * math.convert_like(
            math.cast_like(math.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), gamma_10), gamma_10
        )
        K2 = math.sqrt(gamma_20 + math.eps) * math.convert_like(
            math.cast_like(math.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]), gamma_20), gamma_20
        )
        K3 = math.sqrt(gamma_21 + math.eps) * math.convert_like(
            math.cast_like(math.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), gamma_21), gamma_21
        )
        return [K0, K1, K2, K3]