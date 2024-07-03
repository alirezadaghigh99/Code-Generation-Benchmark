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