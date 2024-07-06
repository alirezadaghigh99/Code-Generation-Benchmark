def get_parameters(
        self, trainable_only=True, operations_only=False, **kwargs
    ):  # pylint:disable=unused-argument
        """Return the parameters incident on the quantum script operations.

        The returned parameters are provided in order of appearance
        on the quantum script.

        Args:
            trainable_only (bool): if True, returns only trainable parameters
            operations_only (bool): if True, returns only the parameters of the
                operations excluding parameters to observables of measurements

        **Example**

        >>> ops = [qml.RX(0.432, 0), qml.RY(0.543, 0),
        ...        qml.CNOT((0,"a")), qml.RX(0.133, "a")]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])

        By default, all parameters are trainable and will be returned:

        >>> qscript.get_parameters()
        [0.432, 0.543, 0.133]

        Setting the trainable parameter indices will result in only the specified
        parameters being returned:

        >>> qscript.trainable_params = [1] # set the second parameter as trainable
        >>> qscript.get_parameters()
        [0.543]

        The ``trainable_only`` argument can be set to ``False`` to instead return
        all parameters:

        >>> qscript.get_parameters(trainable_only=False)
        [0.432, 0.543, 0.133]
        """
        if trainable_only:
            params = []
            for p_idx in self.trainable_params:
                par_info = self.par_info[p_idx]
                if operations_only and isinstance(self[par_info["op_idx"]], MeasurementProcess):
                    continue

                op = par_info["op"]
                op_idx = par_info["p_idx"]
                params.append(op.data[op_idx])
            return params

        # If trainable_only=False, return all parameters
        # This is faster than the above and should be used when indexing `par_info` is not needed
        params = [d for op in self.operations for d in op.data]
        if operations_only:
            return params
        for m in self.measurements:
            if m.obs is not None:
                params.extend(m.obs.data)
        return params

