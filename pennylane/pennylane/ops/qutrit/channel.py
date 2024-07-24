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
        return list(kraus_matrices)

class QutritAmplitudeDamping(Channel):
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
    probabilities for subspaces :math:`(0, 1)`, :math:`(0, 2)`, and :math:`(1, 2)` respectively.

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

class TritFlip(Channel):
    r"""
    Single-qutrit trit flip error channel, used for applying "bit flips" on each qutrit subspace.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \sqrt{1-(p_{01} + p_{02} + p_{12})} \begin{bmatrix}
                1 & 0 & 0  \\
                0 & 1 & 0  \\
                0 & 0 & 1
                \end{bmatrix}

    .. math::
        K_1 = \sqrt{p_{01}}\begin{bmatrix}
                0 & 1 & 0  \\
                1 & 0 & 0  \\
                0 & 0 & 1
                \end{bmatrix}, \quad
        K_2 = \sqrt{p_{02}}\begin{bmatrix}
                0 & 0 & 1  \\
                0 & 1 & 0  \\
                1 & 0 & 0
                \end{bmatrix}, \quad
        K_3 = \sqrt{p_{12}}\begin{bmatrix}
                1 & 0 & 0  \\
                0 & 0 & 1  \\
                0 & 1 & 0
                \end{bmatrix}

    where :math:`p_{01}, p_{02}, p_{12} \in [0, 1]` is the probability of a "trit flip" occurring
    within subspaces (0,1), (0,2), and (1,2) respectively.

    .. note::
        The Kraus operators :math:`\{K_0, K_1, K_2, K_3\}` are adapted from the
        `BitFlip <https://docs.pennylane.ai/en/stable/code/api/pennylane.BitFlip.html>`_ channel's Kraus operators.

        This channel is primarily meant to simulate the misclassification inherent to measurements on some platforms.
        An example of a measurement with misclassification can be seen in [`1 <https://arxiv.org/abs/2309.11303>`_] (Fig 1a).

        To maintain normalization :math:`p_{01} + p_{02} + p_{12} \leq 1`.


    **Details:**

    * Number of wires: 1
    * Number of parameters: 3

    Args:
        p_01 (float): The probability that a :math:`|0 \rangle \leftrightarrow |1 \rangle` trit flip error occurs.
        p_02 (float): The probability that a :math:`|0 \rangle \leftrightarrow |2 \rangle` trit flip error occurs.
        p_12 (float): The probability that a :math:`|1 \rangle \leftrightarrow |2 \rangle` trit flip error occurs.
        wires (Sequence[int] or int): The wire the channel acts on.
        id (str or None): String representing the operation (optional).
    """

    num_params = 3
    num_wires = 1
    grad_method = "F"

    def __init__(self, p_01, p_02, p_12, wires, id=None):
        # Verify input
        ps = (p_01, p_02, p_12)
        for p in ps:
            if not math.is_abstract(p) and not 0.0 <= p <= 1.0:
                raise ValueError("All probabilities must be in the interval [0,1]")
        if not any(math.is_abstract(p) for p in ps):
            if not 0.0 <= sum(ps) <= 1.0:
                raise ValueError("The sum of probabilities must be in the interval [0,1]")

        super().__init__(p_01, p_02, p_12, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p_01, p_02, p_12):  # pylint:disable=arguments-differ
        r"""Kraus matrices representing the TritFlip channel.

        Args:
            p_01 (float): The probability that a :math:`|0 \rangle \leftrightarrow |1 \rangle` trit flip error occurs.
            p_02 (float): The probability that a :math:`|0 \rangle \leftrightarrow |2 \rangle` trit flip error occurs.
            p_12 (float): The probability that a :math:`|1 \rangle \leftrightarrow |2 \rangle` trit flip error occurs.

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.TritFlip.compute_kraus_matrices(0.05, 0.01, 0.10)
        [
        array([ [0.91651514, 0.        , 0.        ],
                [0.        , 0.91651514, 0.        ],
                [0.        , 0.        , 0.91651514]]),
        array([ [0.        , 0.2236068 , 0.       ],
                [0.2236068 , 0.        , 0.       ],
                [0.        , 0.        , 0.2236068]]),
        array([ [0.        , 0.        , 0.1      ],
                [0.        , 0.1       , 0.       ],
                [0.1       , 0.        , 0.       ]]),
        array([ [0.31622777, 0.        , 0.        ],
                [0.        , 0.        , 0.31622777],
                [0.        , 0.31622777, 0.        ]])
        ]
        """
        K0 = math.sqrt(1 - (p_01 + p_02 + p_12) + math.eps) * math.convert_like(
            math.cast_like(np.eye(3), p_01), p_01
        )
        K1 = math.sqrt(p_01 + math.eps) * math.convert_like(
            math.cast_like(math.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), p_01), p_01
        )
        K2 = math.sqrt(p_02 + math.eps) * math.convert_like(
            math.cast_like(math.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), p_02), p_02
        )
        K3 = math.sqrt(p_12 + math.eps) * math.convert_like(
            math.cast_like(math.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), p_12), p_12
        )
        return [K0, K1, K2, K3]

class QutritDepolarizingChannel(Channel):
    r"""
    Single-qutrit symmetrically depolarizing error channel.
    This channel is modelled by the Kraus matrices generated by the following relationship:

    .. math::
        K_0 = K_{0,0} = \sqrt{1-p} \begin{bmatrix}
                1 & 0 & 0\\
                0 & 1 & 0\\
                0 & 0 & 1
                \end{bmatrix}, \quad
        K_{i,j} = \sqrt{\frac{p}{8}}X^iZ^j

    Where:

    .. math::
        X = \begin{bmatrix}
                0 & 1 & 0 \\
                0 & 0 & 1 \\
                1 & 0 & 0
                \end{bmatrix}, \quad
        Z = \begin{bmatrix}
                1 & 0 & 0\\
                0 & \omega & 0\\
                0 & 0 & \omega^2
                \end{bmatrix}


    These relations create the following Kraus matrices:

    .. math::
        \begin{matrix}
            K_0 = K_{0,0} = \sqrt{1-p} \begin{bmatrix}
                1 & 0 & 0\\
                0 & 1 & 0\\
                0 & 0 & 1
                \end{bmatrix}&
            K_1 = K_{0,1} = \sqrt{\frac{p}{8}}\begin{bmatrix}
                1 & 0 & 0\\
                0 & \omega & 0\\
                0 & 0 & \omega^2
                \end{bmatrix}&
            K_2 = K_{0,2} = \sqrt{\frac{p}{8}}\begin{bmatrix}
                1 & 0 & 0\\
                0 & \omega^2 & 0\\
                0 & 0 & \omega
                \end{bmatrix}\\
            K_3 = K_{1,0} = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & 1 & 0 \\
                0 & 0 & 1 \\
                1 & 0 & 0
                \end{bmatrix}&
            K_4 = K_{1,1} = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & \omega & 0 \\
                0 & 0 & \omega^2 \\
                1 & 0 & 0
                \end{bmatrix}&
            K_5 = K_{1,2} = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & \omega^2 & 0 \\
                0 & 0 & \omega \\
                1 & 0 & 0
                \end{bmatrix}\\
            K_6 = K_{2,0} = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & 0 & 1 \\
                1 & 0 & 0 \\
                0 & 1 & 0
                \end{bmatrix}&
            K_7 = K_{2,1} = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & 0 & \omega^2 \\
                1 & 0 & 0 \\
                0 & \omega & 0
                \end{bmatrix}&
            K_8 = K_{2,2} = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & 0 & \omega \\
                1 & 0 & 0 \\
                0 & \omega^2 & 0
                \end{bmatrix}
        \end{matrix}

    Where :math:`\omega=\exp(\frac{2\pi}{3})`  is the third root of unity,
    and :math:`p \in [0, 1]` is the depolarization probability, equally
    divided in the application of all qutrit Pauli operators.

    .. note::

        The Kraus operators :math:`\{K_0 \ldots K_8\}` used are the representations of the single qutrit Pauli group.
        These Pauli group operators are defined in [`1 <https://doi.org/10.48550/arXiv.quant-ph/9802007>`_] (Eq. 5).
        The Kraus Matrices we use are adapted from [`2 <https://doi.org/10.48550/arXiv.1905.10481>`_] (Eq. 5).
        For this definition, please make a note of the following:

        * For :math:`p = 0`, the channel will be an Identity channel, i.e., a noise-free channel.
        * For :math:`p = \frac{8}{9}`, the channel will be a fully depolarizing channel.
        * For :math:`p = 1`, the channel will be a uniform error channel.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        p (float): Each qutrit Pauli operator is applied with probability :math:`\frac{p}{8}`
        wires (Sequence[int] or int): The wire the channel acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1
    grad_method = "A"
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    def __init__(self, p, wires, id=None):
        super().__init__(p, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p):  # pylint:disable=arguments-differ
        r"""Kraus matrices representing the qutrit depolarizing channel.

         Args:
             p (float): Each qutrit Pauli gate is applied with probability :math:`\frac{p}{8}`

         Returns:
             list (array): list of Kraus matrices

         **Example**

         >>> np.round(qml.QutritDepolarizingChannel.compute_kraus_matrices(0.5), 3)
         array([[[ 0.707+0.j   ,  0.   +0.j   ,  0.   +0.j   ],
         [ 0.   +0.j   ,  0.707+0.j   ,  0.   +0.j   ],
         [ 0.   +0.j   ,  0.   +0.j   ,  0.707+0.j   ]],

        [[ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
         [ 0.   +0.j   , -0.125+0.217j,  0.   +0.j   ],
         [ 0.   +0.j   ,  0.   +0.j   , -0.125-0.217j]],

        [[ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
         [ 0.   +0.j   , -0.125-0.217j,  0.   +0.j   ],
         [ 0.   +0.j   ,  0.   +0.j   , -0.125+0.217j]],

        [[ 0.   +0.j   ,  0.25 +0.j   ,  0.   +0.j   ],
         [ 0.   +0.j   ,  0.   +0.j   ,  0.25 +0.j   ],
         [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ]],

        [[ 0.   +0.j   , -0.125+0.217j,  0.   +0.j   ],
         [ 0.   +0.j   ,  0.   +0.j   , -0.125-0.217j],
         [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ]],

        [[ 0.   +0.j   , -0.125-0.217j,  0.   +0.j   ],
         [ 0.   +0.j   ,  0.   +0.j   , -0.125+0.217j],
         [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ]],

        [[ 0.   +0.j   ,  0.   +0.j   ,  0.25 +0.j   ],
         [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
         [ 0.   +0.j   ,  0.25 +0.j   ,  0.   +0.j   ]],

        [[ 0.   +0.j   ,  0.   +0.j   , -0.125-0.217j],
         [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
         [ 0.   +0.j   , -0.125+0.217j,  0.   +0.j   ]],

        [[ 0.   +0.j   ,  0.   +0.j   , -0.125+0.217j],
         [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
         [ 0.   +0.j   , -0.125-0.217j,  0.   +0.j   ]]])
        """
        if not math.is_abstract(p) and not 0.0 <= p <= 1.0:
            raise ValueError("p must be in the interval [0,1]")

        interface = math.get_interface(p)

        w = math.exp(2j * np.pi / 3)
        one = 1
        z = 0

        if interface == "tensorflow":
            p = math.cast_like(p, 1j)
            w = math.cast_like(w, p)
            one = math.cast_like(one, p)
            z = math.cast_like(z, p)

        w2 = w**2

        # The matrices are explicitly written, not generated to ensure PyTorch differentiation.
        depolarizing_mats = [
            [[one, z, z], [z, w, z], [z, z, w2]],
            [[one, z, z], [z, w2, z], [z, z, w]],
            [[z, one, z], [z, z, one], [one, z, z]],
            [[z, w, z], [z, z, w2], [one, z, z]],
            [[z, w2, z], [z, z, w], [one, z, z]],
            [[z, z, one], [one, z, z], [z, one, z]],
            [[z, z, w2], [one, z, z], [z, w, z]],
            [[z, z, w], [one, z, z], [z, w2, z]],
        ]

        normalization = math.sqrt(p / 8 + math.eps)
        Ks = [normalization * math.array(m, like=interface) for m in depolarizing_mats]
        identity = math.sqrt(1 - p + math.eps) * math.array(
            math.eye(QUDIT_DIM, dtype=complex), like=interface
        )

        return [identity] + Ks

class QutritAmplitudeDamping(Channel):
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
    probabilities for subspaces :math:`(0, 1)`, :math:`(0, 2)`, and :math:`(1, 2)` respectively.

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

