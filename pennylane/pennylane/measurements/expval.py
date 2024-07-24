class ExpectationMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the expectation value of the supplied observable.

    Please refer to :func:`pennylane.expval` for detailed documentation.

    Args:
        obs (Union[.Operator, .MeasurementValue]): The observable that is to be measured
            as part of the measurement process. Not all measurement processes require observables
            (for example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    return_type = Expectation

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        if not shots.has_partitioned_shots:
            return ()
        num_shot_elements = sum(s.copies for s in shots.shot_vector)
        return tuple(() for _ in range(num_shot_elements))

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        if not self.wires:
            return qml.math.squeeze(self.eigvals())
        # estimate the ev
        op = self.mv if self.mv is not None else self.obs
        with qml.queuing.QueuingManager.stop_recording():
            samples = SampleMP(
                obs=op,
                eigvals=self._eigvals,
                wires=self.wires if self._eigvals is not None else None,
            ).process_samples(
                samples=samples, wire_order=wire_order, shot_range=shot_range, bin_size=bin_size
            )

        # With broadcasting, we want to take the mean over axis 1, which is the -1st/-2nd with/
        # without bin_size. Without broadcasting, axis 0 is the -1st/-2nd with/without bin_size
        axis = -1 if bin_size is None else -2
        # TODO: do we need to squeeze here? Maybe remove with new return types
        return qml.math.squeeze(qml.math.mean(samples, axis=axis))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        # This also covers statistics for mid-circuit measurements manipulated using
        # arithmetic operators
        # we use ``self.wires`` instead of ``self.obs`` because the observable was
        # already applied to the state
        if not self.wires:
            return qml.math.squeeze(self.eigvals())
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_state(state=state, wire_order=wire_order)
        # In case of broadcasting, `prob` has two axes and this is a matrix-vector product
        return self._calculate_expectation(prob)

    def process_counts(self, counts: dict, wire_order: Wires):
        with qml.QueuingManager.stop_recording():
            probs = qml.probs(wires=self.wires).process_counts(counts=counts, wire_order=wire_order)
        return self._calculate_expectation(probs)

    def _calculate_expectation(self, probabilities):
        """
        Calculate the of expectation set of probabilities.

        Args:
            probabilities (array): the probabilities of collapsing to eigen states
        """
        eigvals = qml.math.asarray(self.eigvals(), dtype="float64")
        return qml.math.dot(probabilities, eigvals)

class ExpectationMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the expectation value of the supplied observable.

    Please refer to :func:`pennylane.expval` for detailed documentation.

    Args:
        obs (Union[.Operator, .MeasurementValue]): The observable that is to be measured
            as part of the measurement process. Not all measurement processes require observables
            (for example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    return_type = Expectation

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        if not shots.has_partitioned_shots:
            return ()
        num_shot_elements = sum(s.copies for s in shots.shot_vector)
        return tuple(() for _ in range(num_shot_elements))

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        if not self.wires:
            return qml.math.squeeze(self.eigvals())
        # estimate the ev
        op = self.mv if self.mv is not None else self.obs
        with qml.queuing.QueuingManager.stop_recording():
            samples = SampleMP(
                obs=op,
                eigvals=self._eigvals,
                wires=self.wires if self._eigvals is not None else None,
            ).process_samples(
                samples=samples, wire_order=wire_order, shot_range=shot_range, bin_size=bin_size
            )

        # With broadcasting, we want to take the mean over axis 1, which is the -1st/-2nd with/
        # without bin_size. Without broadcasting, axis 0 is the -1st/-2nd with/without bin_size
        axis = -1 if bin_size is None else -2
        # TODO: do we need to squeeze here? Maybe remove with new return types
        return qml.math.squeeze(qml.math.mean(samples, axis=axis))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        # This also covers statistics for mid-circuit measurements manipulated using
        # arithmetic operators
        # we use ``self.wires`` instead of ``self.obs`` because the observable was
        # already applied to the state
        if not self.wires:
            return qml.math.squeeze(self.eigvals())
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_state(state=state, wire_order=wire_order)
        # In case of broadcasting, `prob` has two axes and this is a matrix-vector product
        return self._calculate_expectation(prob)

    def process_counts(self, counts: dict, wire_order: Wires):
        with qml.QueuingManager.stop_recording():
            probs = qml.probs(wires=self.wires).process_counts(counts=counts, wire_order=wire_order)
        return self._calculate_expectation(probs)

    def _calculate_expectation(self, probabilities):
        """
        Calculate the of expectation set of probabilities.

        Args:
            probabilities (array): the probabilities of collapsing to eigen states
        """
        eigvals = qml.math.asarray(self.eigvals(), dtype="float64")
        return qml.math.dot(probabilities, eigvals)

