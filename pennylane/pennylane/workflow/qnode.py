def get_gradient_fn(
        device, interface, diff_method="best", tape: Optional["qml.tape.QuantumTape"] = None
    ):
        """Determine the best differentiation method, interface, and device
        for a requested device, interface, and diff method.

        Args:
            device (.Device): PennyLane device
            interface (str): name of the requested interface
            diff_method (str or .TransformDispatcher): The requested method of differentiation.
                If a string, allowed options are ``"best"``, ``"backprop"``, ``"adjoint"``,
                ``"device"``, ``"parameter-shift"``, ``"hadamard"``, ``"finite-diff"``, or ``"spsa"``.
                A gradient transform may also be passed here.
            tape (Optional[.QuantumTape]): the circuit that will be differentiated. Should include shots information.

        Returns:
            tuple[str or .TransformDispatcher, dict, .Device: Tuple containing the ``gradient_fn``,
            ``gradient_kwargs``, and the device to use when calling the execute function.
        """

        config = _make_execution_config(None, diff_method)
        if isinstance(device, qml.devices.Device):
            if device.supports_derivatives(config, circuit=tape):
                new_config = device.preprocess(config)[1]
                return new_config.gradient_method, {}, device
            if diff_method in {"backprop", "adjoint", "device"}:  # device-only derivatives
                raise qml.QuantumFunctionError(
                    f"Device {device} does not support {diff_method} with requested circuit."
                )

        if diff_method == "best":
            return QNode.get_best_method(device, interface, tape=tape)

        if isinstance(device, qml.devices.LegacyDevice):
            # handled by device.supports_derivatives with new device interface
            if diff_method == "backprop":
                return QNode._validate_backprop_method(device, interface, tape=tape)

            if diff_method == "adjoint":
                return QNode._validate_adjoint_method(device)

            if diff_method == "device":
                return QNode._validate_device_method(device)

        if diff_method == "parameter-shift":
            return QNode._validate_parameter_shift(device)

        if diff_method == "finite-diff":
            return qml.gradients.finite_diff, {}, device

        if diff_method == "spsa":
            return qml.gradients.spsa_grad, {}, device

        if diff_method == "hadamard":
            return qml.gradients.hadamard_grad, {}, device

        if isinstance(diff_method, str):
            raise qml.QuantumFunctionError(
                f"Differentiation method {diff_method} not recognized. Allowed "
                "options are ('best', 'parameter-shift', 'backprop', 'finite-diff', "
                "'device', 'adjoint', 'spsa', 'hadamard')."
            )

        if isinstance(diff_method, qml.transforms.core.TransformDispatcher):
            return diff_method, {}, device

        raise qml.QuantumFunctionError(
            f"Differentiation method {diff_method} must be a gradient transform or a string."
        )

