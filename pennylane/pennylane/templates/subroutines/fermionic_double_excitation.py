class FermionicDoubleExcitation(Operation):
    r"""Circuit to exponentiate the tensor product of Pauli matrices representing the
    double-excitation operator entering the Unitary Coupled-Cluster Singles
    and Doubles (UCCSD) ansatz. UCCSD is a VQE ansatz commonly used to run quantum
    chemistry simulations.

    The CC double-excitation operator is given by

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are the fermionic annihilation and
    creation operators and the indices :math:`r, s` and :math:`p, q` run over the occupied and
    unoccupied molecular orbitals, respectively. Using the `Jordan-Wigner transformation
    <https://arxiv.org/abs/1208.5986>`_ the fermionic operator defined above can be written
    in terms of Pauli matrices (for more details see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p + \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.}  ) \Big\}

    The quantum circuit to exponentiate the tensor product of Pauli matrices entering
    the latter equation is shown below (see `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    |

    .. figure:: ../../_static/templates/subroutines/double_excitation_unitary.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    |

    As explained in `Seely et al. (2012) <https://arxiv.org/abs/1208.5986>`_,
    the exponential of a tensor product of Pauli-Z operators can be decomposed in terms of
    :math:`2(n-1)` CNOT gates and a single-qubit Z-rotation referred to as :math:`U_\theta` in
    the figure above. If there are :math:`X` or:math:`Y` Pauli matrices in the product, the
    Hadamard (:math:`H`) or :math:`R_x` gate has to be applied to change to the :math:`X`
    or :math:`Y` basis, respectively. The latter operations are denoted as
    :math:`U_1`, :math:`U_2`, :math:`U_3` and :math:`U_4` in the figure above. See the
    Usage Details section for more details.

    Args:
        weight (float or tensor_like): angle :math:`\theta` entering the Z rotation acting on wire ``p``
        wires1 (Iterable): Wires of the qubits representing the subset of occupied orbitals
            in the interval ``[s, r]``. The first wire is interpreted as ``s``
            and the last wire as ``r``.
            Wires in between are acted on with CNOT gates to compute the parity of the set of qubits.
        wires2 (Iterable): Wires of the qubits representing the subset of unoccupied
            orbitals in the interval ``[q, p]``. The first wire is interpreted as ``q`` and
            the last wire is interpreted as ``p``. Wires in between are acted on with CNOT gates
            to compute the parity of the set of qubits.

    .. details::
        :title: Usage Details

        Notice that:

        #. :math:`\hat{U}_{pqrs}(\theta)` involves eight exponentiations where
           :math:`\hat{U}_1`, :math:`\hat{U}_2`, :math:`\hat{U}_3`, :math:`\hat{U}_4` and
           :math:`\hat{U}_\theta` are defined as follows,

           .. math::

               [U_1, && U_2, U_3, U_4, U_{\theta}] = \\
               && \Bigg\{\bigg[H, H, R_x(-\frac{\pi}{2}), H, R_z(\theta/8)\bigg],
               \bigg[R_x(-\frac{\pi}{2}), H, R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}),
               R_z(\frac{\theta}{8}) \bigg], \\
               && \bigg[H, R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}),
               R_z(\frac{\theta}{8}) \bigg], \bigg[H, H, H, R_x(-\frac{\pi}{2}),
               R_z(\frac{\theta}{8}) \bigg], \\
               && \bigg[R_x(-\frac{\pi}{2}), H, H, H, R_z(-\frac{\theta}{8}) \bigg],
               \bigg[H, R_x(-\frac{\pi}{2}), H, H, R_z(-\frac{\theta}{8}) \bigg], \\
               && \bigg[R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}),
               H, R_z(-\frac{\theta}{8}) \bigg], \bigg[R_x(-\frac{\pi}{2}), R_x(-\frac{\pi}{2}),
               H, R_x(-\frac{\pi}{2}), R_z(-\frac{\theta}{8}) \bigg] \Bigg\}

        #. For a given quadruple ``[s, r, q, p]`` with :math:`p>q>r>s`, seventy-two single-qubit
           and ``16*(len(wires1)-1 + len(wires2)-1 + 1)`` CNOT operations are applied.
           Consecutive CNOT gates act on qubits with indices between ``s`` and ``r`` and
           ``q`` and ``p`` while a single CNOT acts on wires ``r`` and ``q``. The operations
           performed across these qubits are shown in dashed lines in the figure above.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml

            dev = qml.device('default.qubit', wires=5)

            @qml.qnode(dev)
            def circuit(weight, wires1=None, wires2=None):
                qml.FermionicDoubleExcitation(weight, wires1=wires1, wires2=wires2)
                return qml.expval(qml.Z(0))

            weight = 1.34817
            print(circuit(weight, wires1=[0, 1], wires2=[2, 3, 4]))

    """

    num_wires = AnyWires
    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    def _flatten(self):
        return self.data, (self.hyperparameters["wires1"], self.hyperparameters["wires2"])

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @classmethod
    def _unflatten(cls, data, metadata) -> "FermionicDoubleExcitation":
        return cls(data[0], wires1=metadata[0], wires2=metadata[1])

    def __init__(self, weight, wires1=None, wires2=None, id=None):
        if len(wires1) < 2:
            raise ValueError(
                f"expected at least two wires representing the occupied orbitals; "
                f"got {len(wires1)}"
            )
        if len(wires2) < 2:
            raise ValueError(
                f"expected at least two wires representing the unoccupied orbitals; "
                f"got {len(wires2)}"
            )

        shape = qml.math.shape(weight)
        if shape != ():
            raise ValueError(f"Weight must be a scalar; got shape {shape}.")

        wires1 = qml.wires.Wires(wires1)
        wires2 = qml.wires.Wires(wires2)

        self._hyperparameters = {
            "wires1": wires1,
            "wires2": wires2,
        }

        wires = wires1 + wires2
        super().__init__(weight, wires=wires, id=id)

    def map_wires(self, wire_map: dict):
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        for key in ["wires1", "wires2"]:
            new_op._hyperparameters[key] = Wires(
                [wire_map.get(wire, wire) for wire in self._hyperparameters[key]]
            )
        return new_op

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weight, wires, wires1, wires2
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.FermionicDoubleExcitation.decomposition`.

        Args:
            weight (float or tensor_like): angle :math:`\theta` entering the Z rotation
            wires (Any or Iterable[Any]): full set of wires that the operator acts on
            wires1 (Iterable): Wires of the qubits representing the subset of occupied orbitals
                in the interval ``[s, r]``.
            wires2 (Iterable): Wires of the qubits representing the subset of unoccupied
                orbitals in the interval ``[q, p]``.

        Returns:
            list[.Operator]: decomposition of the operator
        """
        s = wires1[0]
        r = wires1[-1]
        q = wires2[0]
        p = wires2[-1]

        # Sequence of the wires entering the CNOTs
        cnots_occ = [wires1[l : l + 2] for l in range(len(wires1) - 1)]
        cnots_unocc = [wires2[l : l + 2] for l in range(len(wires2) - 1)]

        set_cnot_wires = cnots_occ + [[r, q]] + cnots_unocc

        op_list = []

        op_list.extend(_layer1(weight, s, r, q, p, set_cnot_wires))
        op_list.extend(_layer2(weight, s, r, q, p, set_cnot_wires))
        op_list.extend(_layer3(weight, s, r, q, p, set_cnot_wires))
        op_list.extend(_layer4(weight, s, r, q, p, set_cnot_wires))
        op_list.extend(_layer5(weight, s, r, q, p, set_cnot_wires))
        op_list.extend(_layer6(weight, s, r, q, p, set_cnot_wires))
        op_list.extend(_layer7(weight, s, r, q, p, set_cnot_wires))
        op_list.extend(_layer8(weight, s, r, q, p, set_cnot_wires))

        return op_list

