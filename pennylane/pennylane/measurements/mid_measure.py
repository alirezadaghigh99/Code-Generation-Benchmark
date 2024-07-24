class MeasurementValue(Generic[T]):
    """A class representing unknown measurement outcomes in the qubit model.

    Measurements on a single qubit in the computational basis are assumed.

    Args:
        measurements (list[.MidMeasureMP]): The measurement(s) that this object depends on.
        processing_fn (callable): A lazily transformation applied to the measurement values.
    """

    name = "MeasurementValue"

    def __init__(self, measurements, processing_fn):
        self.measurements = measurements
        self.processing_fn = processing_fn

    def _items(self):
        """A generator representing all the possible outcomes of the MeasurementValue."""
        num_meas = len(self.measurements)
        for i in range(2**num_meas):
            branch = tuple(int(b) for b in f"{i:0{num_meas}b}")
            yield branch, self.processing_fn(*branch)

    def _postselected_items(self):
        """A generator representing all the possible outcomes of the MeasurementValue,
        taking postselection into account."""
        # pylint: disable=stop-iteration-return
        ps = {i: p for i, m in enumerate(self.measurements) if (p := m.postselect) is not None}
        num_non_ps = len(self.measurements) - len(ps)
        if num_non_ps == 0:
            yield (), self.processing_fn(*ps.values())
            return
        for i in range(2**num_non_ps):
            # Create the branch ignoring postselected measurements
            non_ps_branch = tuple(int(b) for b in f"{i:0{num_non_ps}b}")
            # We want a consumable iterable and the static tuple above
            non_ps_branch_iter = iter(non_ps_branch)
            # Extend the branch to include postselected measurements
            full_branch = tuple(
                ps[j] if j in ps else next(non_ps_branch_iter)
                for j in range(len(self.measurements))
            )
            # Return the reduced non-postselected branch and the procesing function
            # evaluated on the full branch
            yield non_ps_branch, self.processing_fn(*full_branch)

    @property
    def wires(self):
        """Returns a list of wires corresponding to the mid-circuit measurements."""
        return Wires.all_wires([m.wires for m in self.measurements])

    @property
    def branches(self):
        """A dictionary representing all possible outcomes of the MeasurementValue."""
        ret_dict = {}
        num_meas = len(self.measurements)
        for i in range(2**num_meas):
            branch = tuple(int(b) for b in f"{i:0{num_meas}b}")
            ret_dict[branch] = self.processing_fn(*branch)
        return ret_dict

    def map_wires(self, wire_map):
        """Returns a copy of the current ``MeasurementValue`` with the wires of each measurement changed
        according to the given wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            MeasurementValue: new ``MeasurementValue`` instance with measurement wires mapped
        """
        mapped_measurements = [m.map_wires(wire_map) for m in self.measurements]
        return MeasurementValue(mapped_measurements, self.processing_fn)

    def _transform_bin_op(self, base_bin, other):
        """Helper function for defining dunder binary operations."""
        if isinstance(other, MeasurementValue):
            # pylint: disable=protected-access
            return self._merge(other)._apply(lambda t: base_bin(t[0], t[1]))
        # if `other` is not a MeasurementValue then apply it to each branch
        return self._apply(lambda v: base_bin(v, other))

    def __invert__(self):
        """Return a copy of the measurement value with an inverted control
        value."""
        return self._apply(lambda v: not v)

    def __eq__(self, other):
        return self._transform_bin_op(lambda a, b: a == b, other)

    def __ne__(self, other):
        return self._transform_bin_op(lambda a, b: a != b, other)

    def __add__(self, other):
        return self._transform_bin_op(lambda a, b: a + b, other)

    def __radd__(self, other):
        return self._apply(lambda v: other + v)

    def __sub__(self, other):
        return self._transform_bin_op(lambda a, b: a - b, other)

    def __rsub__(self, other):
        return self._apply(lambda v: other - v)

    def __mul__(self, other):
        return self._transform_bin_op(lambda a, b: a * b, other)

    def __rmul__(self, other):
        return self._apply(lambda v: other * v)

    def __truediv__(self, other):
        return self._transform_bin_op(lambda a, b: a / b, other)

    def __rtruediv__(self, other):
        return self._apply(lambda v: other / v)

    def __lt__(self, other):
        return self._transform_bin_op(lambda a, b: a < b, other)

    def __le__(self, other):
        return self._transform_bin_op(lambda a, b: a <= b, other)

    def __gt__(self, other):
        return self._transform_bin_op(lambda a, b: a > b, other)

    def __ge__(self, other):
        return self._transform_bin_op(lambda a, b: a >= b, other)

    def __and__(self, other):
        return self._transform_bin_op(qml.math.logical_and, other)

    def __or__(self, other):
        return self._transform_bin_op(qml.math.logical_or, other)

    def _apply(self, fn):
        """Apply a post computation to this measurement"""
        return MeasurementValue(self.measurements, lambda *x: fn(self.processing_fn(*x)))

    def concretize(self, measurements: dict):
        """Returns a concrete value from a dictionary of hashes with concrete values."""
        values = tuple(measurements[meas] for meas in self.measurements)
        return self.processing_fn(*values)

    def _merge(self, other: "MeasurementValue"):
        """Merge two measurement values"""

        # create a new merged list with no duplicates and in lexical ordering
        merged_measurements = list(set(self.measurements).union(set(other.measurements)))
        merged_measurements.sort(key=lambda m: m.id)

        # create a new function that selects the correct indices for each sub function
        def merged_fn(*x):
            sub_args_1 = (x[i] for i in [merged_measurements.index(m) for m in self.measurements])
            sub_args_2 = (x[i] for i in [merged_measurements.index(m) for m in other.measurements])

            out_1 = self.processing_fn(*sub_args_1)
            out_2 = other.processing_fn(*sub_args_2)

            return out_1, out_2

        return MeasurementValue(merged_measurements, merged_fn)

    def __getitem__(self, i):
        branch = tuple(int(b) for b in f"{i:0{len(self.measurements)}b}")
        return self.processing_fn(*branch)

    def __str__(self):
        lines = []
        num_meas = len(self.measurements)
        for i in range(2**num_meas):
            branch = tuple(int(b) for b in f"{i:0{num_meas}b}")
            id_branch_mapping = [
                f"{self.measurements[j].id}={branch[j]}" for j in range(len(branch))
            ]
            lines.append(
                "if " + ",".join(id_branch_mapping) + " => " + str(self.processing_fn(*branch))
            )
        return "\n".join(lines)

    def __repr__(self):
        return f"MeasurementValue(wires={self.wires.tolist()})"

class MidMeasureMP(MeasurementProcess):
    """Mid-circuit measurement.

    This class additionally stores information about unknown measurement outcomes in the qubit model.
    Measurements on a single qubit in the computational basis are assumed.

    Please refer to :func:`pennylane.measure` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        reset (bool): Whether to reset the wire after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.
        id (str): Custom label given to a measurement instance.
    """

    def _flatten(self):
        metadata = (("wires", self.raw_wires), ("reset", self.reset), ("id", self.id))
        return (None, None), metadata

    def __init__(
        self,
        wires: Optional[Wires] = None,
        reset: Optional[bool] = False,
        postselect: Optional[int] = None,
        id: Optional[str] = None,
    ):
        self.batch_size = None
        super().__init__(wires=Wires(wires), id=id)
        self.reset = reset
        self.postselect = postselect

    # pylint: disable=arguments-renamed, arguments-differ
    @classmethod
    def _primitive_bind_call(cls, wires=None, reset=False, postselect=None, id=None):
        wires = () if wires is None else wires
        return cls._wires_primitive.bind(*wires, reset=reset, postselect=postselect)

    @classmethod
    def _abstract_eval(
        cls,
        n_wires: Optional[int] = None,
        has_eigvals=False,
        shots: Optional[int] = None,
        num_device_wires: int = 0,
    ) -> tuple:
        return (), int

    def label(self, decimals=None, base_label=None, cache=None):  # pylint: disable=unused-argument
        r"""How the mid-circuit measurement is represented in diagrams and drawings.

        Args:
            decimals=None (Int): If ``None``, no parameters are included. Else,
                how to round the parameters.
            base_label=None (Iterable[str]): overwrite the non-parameter component of the label.
                Must be same length as ``obs`` attribute.
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings
        """
        _label = "┤↗"
        if self.postselect is not None:
            _label += "₁" if self.postselect == 1 else "₀"

        _label += "├" if not self.reset else "│  │0⟩"

        return _label

    @property
    def return_type(self):
        return MidMeasure

    @property
    def samples_computational_basis(self):
        return False

    @property
    def _queue_category(self):
        return "_ops"

    @property
    def hash(self):
        """int: Returns an integer hash uniquely representing the measurement process"""
        fingerprint = (
            self.__class__.__name__,
            tuple(self.wires.tolist()),
            self.id,
        )

        return hash(fingerprint)

    @property
    def data(self):
        """The data of the measurement. Needed to match the Operator API."""
        return []

    @property
    def name(self):
        """The name of the measurement. Needed to match the Operator API."""
        return self.__class__.__name__

class MidMeasureMP(MeasurementProcess):
    """Mid-circuit measurement.

    This class additionally stores information about unknown measurement outcomes in the qubit model.
    Measurements on a single qubit in the computational basis are assumed.

    Please refer to :func:`pennylane.measure` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        reset (bool): Whether to reset the wire after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.
        id (str): Custom label given to a measurement instance.
    """

    def _flatten(self):
        metadata = (("wires", self.raw_wires), ("reset", self.reset), ("id", self.id))
        return (None, None), metadata

    def __init__(
        self,
        wires: Optional[Wires] = None,
        reset: Optional[bool] = False,
        postselect: Optional[int] = None,
        id: Optional[str] = None,
    ):
        self.batch_size = None
        super().__init__(wires=Wires(wires), id=id)
        self.reset = reset
        self.postselect = postselect

    # pylint: disable=arguments-renamed, arguments-differ
    @classmethod
    def _primitive_bind_call(cls, wires=None, reset=False, postselect=None, id=None):
        wires = () if wires is None else wires
        return cls._wires_primitive.bind(*wires, reset=reset, postselect=postselect)

    @classmethod
    def _abstract_eval(
        cls,
        n_wires: Optional[int] = None,
        has_eigvals=False,
        shots: Optional[int] = None,
        num_device_wires: int = 0,
    ) -> tuple:
        return (), int

    def label(self, decimals=None, base_label=None, cache=None):  # pylint: disable=unused-argument
        r"""How the mid-circuit measurement is represented in diagrams and drawings.

        Args:
            decimals=None (Int): If ``None``, no parameters are included. Else,
                how to round the parameters.
            base_label=None (Iterable[str]): overwrite the non-parameter component of the label.
                Must be same length as ``obs`` attribute.
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings
        """
        _label = "┤↗"
        if self.postselect is not None:
            _label += "₁" if self.postselect == 1 else "₀"

        _label += "├" if not self.reset else "│  │0⟩"

        return _label

    @property
    def return_type(self):
        return MidMeasure

    @property
    def samples_computational_basis(self):
        return False

    @property
    def _queue_category(self):
        return "_ops"

    @property
    def hash(self):
        """int: Returns an integer hash uniquely representing the measurement process"""
        fingerprint = (
            self.__class__.__name__,
            tuple(self.wires.tolist()),
            self.id,
        )

        return hash(fingerprint)

    @property
    def data(self):
        """The data of the measurement. Needed to match the Operator API."""
        return []

    @property
    def name(self):
        """The name of the measurement. Needed to match the Operator API."""
        return self.__class__.__name__

