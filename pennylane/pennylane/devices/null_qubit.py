class NullQubit(Device):
    """Null qubit device for PennyLane. This device performs no operations involved in numerical calculations.
    Instead the time spent in execution is dominated by support (or setting up) operations, like tape creation etc.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['aux_wire', 'q1', 'q2']``). Default ``None`` if not specified.
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots
            to use in executions involving this device.

    **Example:**

    .. code-block:: python

        qs = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.CNOT([0, 1])],
            [qml.expval(qml.PauliZ(0)), qml.probs()],
        )
        qscripts = [qs, qs, qs]

    >>> dev = NullQubit()
    >>> program, execution_config = dev.preprocess()
    >>> new_batch, post_processing_fn = program(qscripts)
    >>> results = dev.execute(new_batch, execution_config=execution_config)
    >>> post_processing_fn(results)
    ((array(0.), array([1., 0., 0., 0.])),
     (array(0.), array([1., 0., 0., 0.])),
     (array(0.), array([1., 0., 0., 0.])))


    This device currently supports (trivial) derivatives:

    >>> from pennylane.devices import ExecutionConfig
    >>> dev.supports_derivatives(ExecutionConfig(gradient_method="device"))
    True

    This device can be used to track resource usage:

    .. code-block:: python

        n_layers = 50
        n_wires = 100
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)

        @qml.qnode(dev)
        def circuit(params):
            qml.StronglyEntanglingLayers(params, wires=range(n_wires))
            return [qml.expval(qml.Z(i)) for i in range(n_wires)]

        params = np.random.random(shape)

        with qml.Tracker(dev) as tracker:
            circuit(params)

    >>> tracker.history["resources"][0]
    wires: 100
    gates: 10000
    depth: 502
    shots: Shots(total=None)
    gate_types:
    {'Rot': 5000, 'CNOT': 5000}
    gate_sizes:
    {1: 5000, 2: 5000}


    .. details::
        :title: Tracking

        ``NullQubit`` tracks:

        * ``executions``: the number of unique circuits that would be required on quantum hardware
        * ``shots``: the number of shots
        * ``resources``: the :class:`~.resource.Resources` for the executed circuit.
        * ``simulations``: the number of simulations performed. One simulation can cover multiple QPU executions, such as for non-commuting measurements and batched parameters.
        * ``batches``: The number of times :meth:`~.execute` is called.
        * ``results``: The results of each call of :meth:`~.execute`
        * ``derivative_batches``: How many times :meth:`~.compute_derivatives` is called.
        * ``execute_and_derivative_batches``: How many times :meth:`~.execute_and_compute_derivatives` is called
        * ``vjp_batches``: How many times :meth:`~.compute_vjp` is called
        * ``execute_and_vjp_batches``: How many times :meth:`~.execute_and_compute_vjp` is called
        * ``jvp_batches``: How many times :meth:`~.compute_jvp` is called
        * ``execute_and_jvp_batches``: How many times :meth:`~.execute_and_compute_jvp` is called
        * ``derivatives``: How many circuits are submitted to :meth:`~.compute_derivatives` or :meth:`~.execute_and_compute_derivatives`.
        * ``vjps``: How many circuits are submitted to :meth:`~.compute_vjp` or :meth:`~.execute_and_compute_vjp`
        * ``jvps``: How many circuits are submitted to :meth:`~.compute_jvp` or :meth:`~.execute_and_compute_jvp`

    """

    @property
    def name(self):
        """The name of the device."""
        return "null.qubit"

    def __init__(self, wires=None, shots=None) -> None:
        super().__init__(wires=wires, shots=shots)
        self._debugger = None

    def _simulate(self, circuit, interface):
        shots = circuit.shots
        obj_with_wires = self if self.wires else circuit
        results = tuple(
            zero_measurement(mp, obj_with_wires, shots, circuit.batch_size, interface)
            for mp in circuit.measurements
        )
        if len(results) == 1:
            return results[0]
        if shots.has_partitioned_shots:
            return tuple(zip(*results))
        return results

    def _derivatives(self, circuit, interface):
        shots = circuit.shots
        obj_with_wires = self if self.wires else circuit
        n = len(circuit.trainable_params)
        derivatives = tuple(
            (
                math.zeros_like(
                    zero_measurement(mp, obj_with_wires, shots, circuit.batch_size, interface)
                ),
            )
            * n
            for mp in circuit.measurements
        )
        if n == 1:
            derivatives = tuple(d[0] for d in derivatives)
        return derivatives[0] if len(derivatives) == 1 else derivatives

    @staticmethod
    def _vjp(circuit, interface):
        batch_size = circuit.batch_size
        n = len(circuit.trainable_params)
        res_shape = (n,) if batch_size is None else (n, batch_size)
        return math.zeros(res_shape, like=interface)

    @staticmethod
    def _jvp(circuit, interface):
        jvps = (math.asarray(0.0, like=interface),) * len(circuit.measurements)
        return jvps[0] if len(jvps) == 1 else jvps

    @staticmethod
    def _setup_execution_config(execution_config: ExecutionConfig) -> ExecutionConfig:
        """No-op function to allow for borrowing DefaultQubit.preprocess without AttributeErrors"""
        return execution_config

    @property
    def _max_workers(self):
        """No-op property to allow for borrowing DefaultQubit.preprocess without AttributeErrors"""
        return None

    # pylint: disable=cell-var-from-loop
    def preprocess(
        self, execution_config=DefaultExecutionConfig
    ) -> Tuple[TransformProgram, ExecutionConfig]:
        program, _ = DefaultQubit.preprocess(self, execution_config)
        for t in program:
            if t.transform == decompose.transform:
                original_stopping_condition = t.kwargs["stopping_condition"]

                def new_stopping_condition(op):
                    return (not op.has_decomposition) or original_stopping_condition(op)

                t.kwargs["stopping_condition"] = new_stopping_condition

                original_shots_stopping_condition = t.kwargs.get("stopping_condition_shots", None)
                if original_shots_stopping_condition:

                    def new_shots_stopping_condition(op):
                        return (not op.has_decomposition) or original_shots_stopping_condition(op)

                    t.kwargs["stopping_condition_shots"] = new_shots_stopping_condition

        updated_values = {}
        if execution_config.gradient_method in ["best", "adjoint"]:
            updated_values["gradient_method"] = "device"
        if execution_config.use_device_gradient is None:
            updated_values["use_device_gradient"] = execution_config.gradient_method in {
                "best",
                "device",
                "adjoint",
                "backprop",
            }
        if execution_config.use_device_jacobian_product is None:
            updated_values["use_device_jacobian_product"] = (
                execution_config.gradient_method == "device"
            )
        if execution_config.grad_on_execution is None:
            updated_values["grad_on_execution"] = execution_config.gradient_method == "device"
        return program, replace(execution_config, **updated_values)

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug(
                """Entry with args=(circuits=%s) called by=%s""",
                circuits,
                "::L".join(
                    str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
                ),
            )

        return tuple(
            self._simulate(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )

    def supports_derivatives(self, execution_config=None, circuit=None):
        return execution_config is None or execution_config.gradient_method in (
            "device",
            "backprop",
            "adjoint",
        )

    def supports_vjp(self, execution_config=None, circuit=None):
        return execution_config is None or execution_config.gradient_method in (
            "device",
            "backprop",
            "adjoint",
        )

    def supports_jvp(self, execution_config=None, circuit=None):
        return execution_config is None or execution_config.gradient_method in (
            "device",
            "backprop",
            "adjoint",
        )

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        return tuple(
            self._derivatives(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        results = tuple(
            self._simulate(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )
        jacs = tuple(
            self._derivatives(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )

        return results, jacs

    def compute_jvp(
        self,
        circuits: QuantumTape_or_Batch,
        tangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        return tuple(self._jvp(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits)

    def execute_and_compute_jvp(
        self,
        circuits: QuantumTape_or_Batch,
        tangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        results = tuple(
            self._simulate(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )
        jvps = tuple(self._jvp(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits)

        return results, jvps

    def compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        return tuple(self._vjp(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits)

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        results = tuple(
            self._simulate(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )
        vjps = tuple(self._vjp(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits)
        return results, vjps

