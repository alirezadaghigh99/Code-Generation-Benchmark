class RZ(Operation):
    r"""
    The single qubit Z rotation

    .. math:: R_z(\phi) = e^{-i\phi\sigma_z/2} = \begin{bmatrix}
                e^{-i\phi/2} & 0 \\
                0 & e^{i\phi/2}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_z(\phi)) = \frac{1}{2}\left[f(R_z(\phi+\pi/2)) - f(R_z(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_z(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Z"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Hamiltonian([-0.5], [PauliZ(wires=self.wires)])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.RZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.RZ.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9689+0.2474j]])
        """
        if qml.math.get_interface(theta) == "tensorflow":
            p = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            z = qml.math.zeros_like(p)

            return qml.math.stack([stack_last([p, z]), stack_last([z, qml.math.conj(p)])], axis=-2)

        signs = qml.math.array([-1, 1], like=theta)
        arg = 0.5j * theta

        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))

        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(2, like=diags), diags)

    @staticmethod
    def compute_eigvals(theta):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.RZ.eigvals`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.RZ.compute_eigvals(torch.tensor(0.5))
        tensor([0.9689-0.2474j, 0.9689+0.2474j])
        """
        if qml.math.get_interface(theta) == "tensorflow":
            phase = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            return qml.math.stack([phase, qml.math.conj(phase)], axis=-1)

        prefactors = qml.math.array([-0.5j, 0.5j], like=theta)
        if qml.math.ndim(theta) == 0:
            product = theta * prefactors
        else:
            product = qml.math.outer(theta, prefactors)
        return qml.math.exp(product)

    def adjoint(self):
        return RZ(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [RZ(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        return qml.CRZ(*self.parameters, wires=wire + self.wires)

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)

        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        return RZ(theta, wires=self.wires)

    def single_qubit_rot_angles(self):
        # RZ(\theta) = RZ(\theta) RY(0) RZ(0)
        return [self.data[0], 0.0, 0.0]class RY(Operation):
    r"""
    The single qubit Y rotation

    .. math:: R_y(\phi) = e^{-i\phi\sigma_y/2} = \begin{bmatrix}
                \cos(\phi/2) & -\sin(\phi/2) \\
                \sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_y(\phi)) = \frac{1}{2}\left[f(R_y(\phi+\pi/2)) - f(R_y(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_y(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "Y"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return qml.Hamiltonian([-0.5], [PauliY(wires=self.wires)])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.RY.matrix`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.RY.compute_matrix(torch.tensor(0.5))
        tensor([[ 0.9689, -0.2474],
                [ 0.2474,  0.9689]])
        """

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)
        if qml.math.get_interface(theta) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
        # The following avoids casting an imaginary quantity to reals when backpropagating
        c = (1 + 0j) * c
        s = (1 + 0j) * s
        return qml.math.stack([stack_last([c, -s]), stack_last([s, c])], axis=-2)

    def adjoint(self):
        return RY(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [RY(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        return qml.CRY(*self.parameters, wires=wire + self.wires)

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)

        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        return RY(theta, wires=self.wires)

    def single_qubit_rot_angles(self):
        # RY(\theta) = RZ(0) RY(\theta) RZ(0)
        return [0.0, self.data[0], 0.0]class Rot(Operation):
    r"""
    Arbitrary single qubit rotation

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Number of dimensions per parameter: (0, 0, 0)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R(\phi, \theta, \omega)) = \frac{1}{2}\left[f(R(\phi+\pi/2, \theta, \omega)) - f(R(\phi-\pi/2, \theta, \omega))\right]`
      where :math:`f` is an expectation value depending on :math:`R(\phi, \theta, \omega)`.
      This gradient recipe applies for each angle argument :math:`\{\phi, \theta, \omega\}`.

    .. note::

        If the ``Rot`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.RZ` and :class:`~.RY` gates.

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Any, Wires): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 3
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,), (1,), (1,)]

    def __init__(self, phi, theta, omega, wires, id=None):
        super().__init__(phi, theta, omega, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi, theta, omega):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Rot.matrix`


        Args:
            phi (tensor_like or float): first rotation angle
            theta (tensor_like or float): second rotation angle
            omega (tensor_like or float): third rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.Rot.compute_matrix(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
        tensor([[ 0.9752-0.1977j, -0.0993+0.0100j],
                [ 0.0993+0.0100j,  0.9752+0.1977j]])

        """
        # It might be that they are in different interfaces, e.g.,
        # Rot(0.2, 0.3, tf.Variable(0.5), wires=0)
        # So we need to make sure the matrix comes out having the right type
        interface = qml.math.get_interface(phi, theta, omega)

        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        # If anything is not tensorflow, it has to be casted and then
        if interface == "tensorflow":
            phi = qml.math.cast_like(qml.math.asarray(phi, like=interface), 1j)
            omega = qml.math.cast_like(qml.math.asarray(omega, like=interface), 1j)
            c = qml.math.cast_like(qml.math.asarray(c, like=interface), 1j)
            s = qml.math.cast_like(qml.math.asarray(s, like=interface), 1j)

        # The following variable is used to assert the all terms to be stacked have same shape
        one = qml.math.ones_like(phi) * qml.math.ones_like(omega)
        c = c * one
        s = s * one

        mat = [
            [
                qml.math.exp(-0.5j * (phi + omega)) * c,
                -qml.math.exp(0.5j * (phi - omega)) * s,
            ],
            [
                qml.math.exp(-0.5j * (phi - omega)) * s,
                qml.math.exp(0.5j * (phi + omega)) * c,
            ],
        ]

        return qml.math.stack([stack_last(row) for row in mat], axis=-2)

    @staticmethod
    def compute_decomposition(phi, theta, omega, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.Rot.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            theta (float): rotation angle :math:`\theta`
            omega (float): rotation angle :math:`\omega`
            wires (Any, Wires): the wire the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.Rot.compute_decomposition(1.2, 2.3, 3.4, wires=0)
        [RZ(1.2, wires=[0]), RY(2.3, wires=[0]), RZ(3.4, wires=[0])]

        """
        return [
            RZ(phi, wires=wires),
            RY(theta, wires=wires),
            RZ(omega, wires=wires),
        ]

    def adjoint(self):
        phi, theta, omega = self.parameters
        return Rot(-omega, -theta, -phi, wires=self.wires)

    def _controlled(self, wire):
        return qml.CRot(*self.parameters, wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        return self.data

    def simplify(self):
        """Simplifies into single-rotation gates or a Hadamard if possible.

        >>> qml.Rot(np.pi / 2, 0.1, -np.pi / 2, wires=0).simplify()
        RX(0.1, wires=[0])
        >>> qml.Rot(np.pi, np.pi/2, 0, 0).simplify()
        Hadamard(wires=[0])

        """
        p0, p1, p2 = [p % (4 * np.pi) for p in self.data]

        if _can_replace(p0, 0) and _can_replace(p1, 0) and _can_replace(p2, 0):
            return qml.Identity(wires=self.wires)
        if _can_replace(p0, np.pi / 2) and _can_replace(p2, 7 * np.pi / 2):
            return RX(p1, wires=self.wires)
        if _can_replace(p0, 0) and _can_replace(p2, 0):
            return RY(p1, wires=self.wires)
        if _can_replace(p1, 0):
            return RZ((p0 + p2) % (4 * np.pi), wires=self.wires)
        if _can_replace(p0, np.pi) and _can_replace(p1, np.pi / 2) and _can_replace(p2, 0):
            return Hadamard(wires=self.wires)

        return Rot(p0, p1, p2, wires=self.wires)