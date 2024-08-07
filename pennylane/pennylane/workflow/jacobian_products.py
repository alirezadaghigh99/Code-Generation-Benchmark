class TransformJacobianProducts(JacobianProductCalculator):
    """Compute VJPs, JVPs and Jacobians via a gradient transform :class:`~.TransformDispatcher`.

    Args:
        inner_execute (Callable[[Tuple[QuantumTape]], ResultBatch]): a function that
            executes the batch of circuits and returns their results.
        gradient_transform (.TransformDispatcher): the gradient transform to use.
        gradient_kwargs (dict): Any keyword arguments for the gradient transform.

    Keyword Args:
        cache_full_jacobian=False (bool): Whether or not to compute the full jacobian and cache it,
            instead of treating each call as independent. This keyword argument is used to patch problematic
            autograd behaviour when caching is turned off. In this case, caching will be based on the identity
            of the batch, rather than the potentially expensive :attr:`~.QuantumScript.hash` that is used
            by the cache.

    >>> inner_execute = qml.device('default.qubit').execute
    >>> gradient_transform = qml.gradients.param_shift
    >>> kwargs = {"broadcast": True}
    >>> jpc = TransformJacobianProducts(inner_execute, gradient_transform, kwargs)

    """

    def __repr__(self):
        return (
            f"TransformJacobianProducts({self._inner_execute}, gradient_transform={self._gradient_transform}, "
            f"gradient_kwargs={self._gradient_kwargs}, cache_full_jacobian={self._cache_full_jacobian})"
        )

    def __init__(
        self,
        inner_execute: Callable,
        gradient_transform: "pennylane.transforms.core.TransformDispatcher",
        gradient_kwargs: Optional[dict] = None,
        cache_full_jacobian: bool = False,
    ):
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug(
                "TransformJacobianProduct being created with (%s, %s, %s, %s)",
                (
                    inspect.getsource(inner_execute)
                    if (
                        logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(inner_execute)
                    )
                    else inner_execute
                ),
                gradient_transform,
                gradient_kwargs,
                cache_full_jacobian,
            )
        self._inner_execute = inner_execute
        self._gradient_transform = gradient_transform
        self._gradient_kwargs = gradient_kwargs or {}
        self._cache_full_jacobian = cache_full_jacobian
        self._cache = LRUCache(maxsize=10)

    def execute_and_compute_jvp(self, tapes: Batch, tangents: Tuple[Tuple[TensorLike]]):
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug("execute_and_compute_jvp called with (%s, %s)", tapes, tangents)

        num_result_tapes = len(tapes)

        if self._cache_full_jacobian:
            jacs = self.compute_jacobian(tapes)
            jvps = _compute_jvps(jacs, tangents, tapes)
            return self._inner_execute(tapes), jvps
        jvp_tapes, jvp_processing_fn = qml.gradients.batch_jvp(
            tapes, tangents, self._gradient_transform, gradient_kwargs=self._gradient_kwargs
        )

        full_batch = tapes + tuple(jvp_tapes)

        full_results = self._inner_execute(full_batch)

        results = full_results[:num_result_tapes]
        jvp_results = full_results[num_result_tapes:]
        jvps = jvp_processing_fn(jvp_results)
        return tuple(results), tuple(jvps)

    def compute_vjp(self, tapes: Batch, dy: Tuple[Tuple[TensorLike]]):
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug("compute_vjp called with (%s, %s)", tapes, dy)

        if self._cache_full_jacobian:
            jacs = self.compute_jacobian(tapes)
            return _compute_vjps(jacs, dy, tapes)

        vjp_tapes, processing_fn = qml.gradients.batch_vjp(
            tapes, dy, self._gradient_transform, gradient_kwargs=self._gradient_kwargs
        )

        vjp_results = self._inner_execute(tuple(vjp_tapes))
        return tuple(processing_fn(vjp_results))

    def execute_and_compute_jacobian(self, tapes: Batch):
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug("execute_and_compute_jacobian called with %s", tapes)

        num_result_tapes = len(tapes)

        jac_tapes, jac_postprocessing = self._gradient_transform(tapes, **self._gradient_kwargs)

        full_batch = tapes + tuple(jac_tapes)
        full_results = self._inner_execute(full_batch)
        results = full_results[:num_result_tapes]
        jac_results = full_results[num_result_tapes:]
        jacs = jac_postprocessing(jac_results)
        return tuple(results), tuple(jacs)

    def compute_jacobian(self, tapes: Batch):
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug("compute_jacobian called with %s", tapes)
        if tapes in self._cache:
            return self._cache[tapes]
        jac_tapes, batch_post_processing = self._gradient_transform(tapes, **self._gradient_kwargs)
        results = self._inner_execute(jac_tapes)
        jacs = tuple(batch_post_processing(results))
        self._cache[tapes] = jacs
        return jacs

class DeviceDerivatives(JacobianProductCalculator):
    """Calculate jacobian products via a device provided jacobian.  This class relies on either ``qml.Device.gradients`` or
    ``qml.devices.Device.compute_derivatives``.

    Args:

        device (Union[pennylane.Device, pennylane.devices.Device]): the device for execution and derivatives.
            Must support first order gradients with the requested configuration.
        execution_config (pennylane.devices.ExecutionConfig): a datastructure containing the options needed to fully
           describe the execution. Only used with :class:`pennylane.devices.Device` from the new device interface.
        gradient_kwargs (dict): a dictionary of keyword arguments for the gradients. Only used with a :class:`~.pennylane.Device`
            from the old device interface.

    **Examples:**

    >>> device = qml.device('default.qubit')
    >>> config = qml.devices.ExecutionConfig(gradient_method="adjoint")
    >>> jpc = DeviceDerivatives(device, config, {})

    This same class can also be used with the old device interface.

    >>> device = qml.device('lightning.qubit', wires=5)
    >>> gradient_kwargs = {"method": "adjoint_jacobian"}
    >>> jpc_lightning = DeviceDerivatives(device, gradient_kwargs=gradient_kwargs)

    **Technical comments on caching and calculating the gradients on execution:**

    In order to store results and Jacobians for the backward pass during the forward pass,
    the ``_jacs_cache`` and ``_results_cache`` properties are ``LRUCache`` objects with a maximum size of 10.
    In the current execution pipeline, only one batch will be used per instance, but a size of 10 adds some extra
    flexibility for future uses.

    Note that batches of identically looking :class:`~.QuantumScript` s that are different instances will be cached separately.
    This is because the ``hash`` of  :class:`~.QuantumScript` is expensive, as it requires inspecting all its constituents,
    which is not worth the effort in this case.

    When a forward pass with :meth:`~.execute_and_cache_jacobian` is called, both the results and the jacobian for the object are stored.

    >>> tape = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.Z(0))])
    >>> batch = (tape, )
    >>> with device.tracker:
    ...     results = jpc.execute_and_cache_jacobian(batch )
    >>> results
    (0.5403023058681398,)
    >>> device.tracker.totals
    {'execute_and_derivative_batches': 1, 'executions': 1, 'derivatives': 1}
    >>> jpc._jacs_cache
    LRUCache({5660934048: (array(-0.84147098),)}, maxsize=10, currsize=1)

    Then when the vjp, jvp, or jacobian is requested, that cached value is used instead of requesting from
    the device again.

    >>> with device.tracker:
    ...     vjp = jpc.compute_vjp(batch , (0.5, ) )
    >>> vjp
    (array([-0.42073549]),)
    >>> device.tracker.totals
    {}

    """

    def __repr__(self):
        return f"<DeviceDerivatives: {self._device.name}, {self._gradient_kwargs}, {self._execution_config}>"

    def __init__(
        self,
        device: Union["qml.devices.Device", "qml.Device"],
        execution_config: Optional["qml.devices.ExecutionConfig"] = None,
        gradient_kwargs: dict = None,
    ):
        if gradient_kwargs is None:
            gradient_kwargs = {}
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug(
                "DeviceDerivatives created with (%s, %s, %s)",
                device,
                execution_config,
                gradient_kwargs,
            )

        self._device = device
        self._execution_config = execution_config
        self._gradient_kwargs = gradient_kwargs

        self._uses_new_device = not isinstance(device, qml.devices.LegacyDevice)

        # only really need to keep most recent entry, but keeping 10 around just in case
        self._results_cache = LRUCache(maxsize=10)
        self._jacs_cache = LRUCache(maxsize=10)

    def _dev_execute_and_compute_derivatives(self, tapes: Batch):
        """
        Converts tapes to numpy before computing the the results and derivatives on the device.

        Dispatches between the two different device interfaces.
        """
        numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(tapes)
        if self._uses_new_device:
            return self._device.execute_and_compute_derivatives(numpy_tapes, self._execution_config)
        return self._device.execute_and_gradients(numpy_tapes, **self._gradient_kwargs)

    def _dev_execute(self, tapes: Batch):
        """
        Converts tapes to numpy before computing just the results on the device.

        Dispatches between the two different device interfaces.
        """
        numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(tapes)
        if self._uses_new_device:
            return self._device.execute(numpy_tapes, self._execution_config)
        return self._device.batch_execute(numpy_tapes)

    def _dev_compute_derivatives(self, tapes: Batch):
        """
        Converts tapes to numpy before computing the derivatives on the device.

        Dispatches between the two different device interfaces.
        """
        numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(tapes)
        if self._uses_new_device:
            return self._device.compute_derivatives(numpy_tapes, self._execution_config)
        return self._device.gradients(numpy_tapes, **self._gradient_kwargs)

    def execute_and_cache_jacobian(self, tapes: Batch):
        """Forward pass used to cache the results and jacobians.

        Args:
            tapes (tuple[`~.QuantumScript`]): the batch of tapes to execute and take derivatives of

        Returns:
            ResultBatch: the results of the execution.

        Side Effects:
            Caches both the results and jacobian into ``_results_cache`` and ``_jacs_cache``.

        """
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug("Forward pass called with %s", tapes)
        results, jac = self._dev_execute_and_compute_derivatives(tapes)
        self._results_cache[tapes] = results
        self._jacs_cache[tapes] = jac
        return results

    def execute_and_compute_jvp(self, tapes: Batch, tangents):
        """Calculate both the results for a batch of tapes and the jvp.

        This method is required to compute JVPs in the JAX interface.

        Args:
            tapes (tuple[`~.QuantumScript`]): The batch of tapes to take the derivatives of
            tangents (Sequence[Sequence[TensorLike]]): the tangents for the parameters of the tape.
                The ``i`` th tangent corresponds to the ``i`` th tape, and the ``j`` th entry into a
                tangent entry corresponds to the ``j`` th trainable parameter of the tape.

        Returns:
            ResultBatch, TensorLike: the results of the execution and the jacobian vector product

        Side Effects:
            caches newly computed results or jacobians if they were not already cached.

        **Examples:**

        For an instance of :class:`~.DeviceDerivatives` ``jpc``, we have:

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.Z(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.Z(0))])
        >>> batch = (tape0, tape1)
        >>> tangents0 = (1.5, )
        >>> tangents1 = (2.0, )
        >>> tangents = (tangents0, tangents1)
        >>> results, jvps = jpc.execute_and_compute_jvp(batch, tangents)
        >>> expected_results = (np.cos(0.1), np.cos(0.2))
        >>> qml.math.allclose(results, expected_results)
        True
        >>> jvps
        (array(-0.14975012), array(-0.39733866))
        >>> expected_jvps = 1.5 * -np.sin(0.1), 2.0 * -np.sin(0.2)
        >>> qml.math.allclose(jvps, expected_jvps)
        True

        While this method could support non-scalar parameters in theory, no implementation currently supports
        jacobians with non-scalar parameters.

        """
        results, jacs = self.execute_and_compute_jacobian(tapes)
        jvps = _compute_jvps(jacs, tangents, tapes)
        return results, jvps

    def compute_vjp(self, tapes, dy):
        """Compute the vjp for a given batch of tapes.

        This method is used by autograd, torch, and tensorflow to compute VJPs.

        Args:
            tapes (tuple[`~.QuantumScript`]): the batch of tapes to take the derivatives of
            dy (tuple[tuple[TensorLike]]): the derivatives of the results of an execution.
                The ``i`` th entry (cotangent) corresponds to the ``i`` th tape, and the ``j`` th entry of the ``i`` th
                cotangent corresponds to the ``j`` th return value of the ``i`` th tape.

        Returns:
            TensorLike: the vector jacobian product.

        Side Effects:
            caches the newly computed jacobian if it wasn't already present in the cache.

        **Examples:**

        For an instance of :class:`~.DeviceDerivatives` ``jpc``, we have:

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.Z(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.Z(0)), qml.expval(qml.X(0))])
        >>> batch = (tape0, tape1)
        >>> dy0 = (0.5, )
        >>> dy1 = (2.0, 3.0)
        >>> dys = (dy0, dy1)
        >>> vjps = jpc.compute_vjp(batch, dys)
        >>> vjps
        (array([-0.04991671]), array([2.54286107]))
        >>> expected_vjp0 = 0.5 * -np.sin(0.1)
        >>> qml.math.allclose(vjps[0], expected_vjp0)
        True
        >>> expected_jvp1 = 2.0 * -np.sin(0.2) + 3.0 * np.cos(0.2)
        >>> qml.math.allclose(vjps[1], expected_vjp1)
        True

        While this method could support non-scalar parameters in theory, no implementation currently supports
        jacobians with non-scalar parameters.

        """
        if tapes in self._jacs_cache:
            if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
                logger.debug(" %s : Retrieving jacobian from cache.", self)
            jacs = self._jacs_cache[tapes]
        else:
            jacs = self._dev_compute_derivatives(tapes)
            self._jacs_cache[tapes] = jacs

        return _compute_vjps(jacs, dy, tapes)

    def compute_jacobian(self, tapes):
        """Compute the full Jacobian for a batch of tapes.

        This method is required to compute Jacobians in the ``jax-jit`` interface

        Args:
            tapes: the batch of tapes to take the Jacobian of

        Returns:
            TensorLike: the full jacobian

        Side Effects:
            caches the newly computed jacobian if it wasn't already present in the cache.

        **Examples:**

        For an instance of :class:`~.DeviceDerivatives` ``jpc``, we have:

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.Z(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.Z(0)), qml.expval(qml.X(0))])
        >>> batch = (tape0, tape1)
        >>> jpc.compute_jacobian(batch)
        (array(-0.09983342), (array(-0.19866933), array(0.98006658)))

        While this method could support non-scalar parameters in theory, no implementation currently supports
        jacobians with non-scalar parameters.

        """
        if tapes in self._jacs_cache:
            if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
                logger.debug("%s : Retrieving jacobian from cache.", self)
            return self._jacs_cache[tapes]

        jacs = self._dev_compute_derivatives(tapes)
        self._jacs_cache[tapes] = jacs
        return jacs

    def execute_and_compute_jacobian(self, tapes):
        if tapes not in self._results_cache and tapes not in self._jacs_cache:
            results, jacs = self._dev_execute_and_compute_derivatives(tapes)
            self._results_cache[tapes] = results
            self._jacs_cache[tapes] = jacs
            return results, jacs

        if tapes not in self._jacs_cache:
            # Here the jac was not cached but the results were. This can not happen because results are never
            # cached alone (note that in the else clause below computing only results, jac must already be present)
            raise NotImplementedError(
                "No path to cache results without caching jac. This branch should not occur."
            )
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug("%s : Retrieving jacobian from cache.", self)
        jacs = self._jacs_cache[tapes]

        if tapes in self._results_cache:
            if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
                logger.debug("%s : Retrieving results from cache.", self)
            results = self._results_cache[tapes]
        else:
            results = self._dev_execute(tapes)
            self._results_cache[tapes] = results

        return results, jacs

class LightningVJPs(DeviceDerivatives):
    """Calculates VJPs natively using lightning.qubit.

    Args:
        device (LightningBase): Lightning ecosystem devices ``lightning.gpu`` or ``lightning.kokkos``.
        gradient_kwargs (Optional[dict]):  Any gradient options.

    >>> dev = qml.device('lightning.qubit', wires=5)
    >>> jpc = LightningVJPs(dev, gradient_kwargs={"use_device_state": True, "method": "adjoint_jacobian"})
    >>> tape = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.Z(0))])
    >>> dev.batch_execute((tape,))
    [array(0.36235775)]
    >>> jpc.compute_vjp((tape,), (0.5,) )
    ((array(-0.46601954),),)
    >>> -0.5 * np.sin(1.2)
    -0.46601954298361314

    """

    def __repr__(self):
        long_to_short_name = {
            "LightningQubit": "lightning.qubit",
            "LightningKokkos": "lightning.kokkos",
            "LightningGPU": "lightning.gpu",
        }
        return f"<LightningVJPs: {long_to_short_name[type(self._device).__name__]}, {self._gradient_kwargs}>"

    def __init__(self, device, gradient_kwargs=None):
        super().__init__(device, gradient_kwargs=gradient_kwargs)
        self._processed_gradient_kwargs = {
            key: value for key, value in self._gradient_kwargs.items() if key != "method"
        }

    def compute_vjp(self, tapes, dy):  # pragma: no cover
        if not all(
            isinstance(m, qml.measurements.ExpectationMP) for t in tapes for m in t.measurements
        ):
            raise NotImplementedError("Lightning device VJPs only support expectation values.")
        results = []
        numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(tapes)
        for dyi, tape in zip(dy, numpy_tapes):
            if len(tape.measurements) == 1:
                dyi = (dyi,)
            dyi = np.array(qml.math.unwrap(dyi))
            if qml.math.ndim(dyi) > 1:
                raise NotImplementedError(
                    "Lightning device VJPs are not supported with jax jacobians."
                )
            vjp_f = self._device.vjp(tape.measurements, dyi, **self._processed_gradient_kwargs)
            out = vjp_f(tape)
            if len(tape.trainable_params) == 1:
                out = (out,)
            results.append(out)
        return tuple(results)

