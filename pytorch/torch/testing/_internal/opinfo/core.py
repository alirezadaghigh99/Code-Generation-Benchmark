class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "input",
        "args",
        "kwargs",
        "output_process_fn_grad",
        "broadcasts_input",
        "name",
    ]

    def __init__(
        self,
        input,
        *var_args,
        args=None,
        kwargs=None,
        output_process_fn_grad=None,
        broadcasts_input=None,
        name=None,
        **var_kwargs,
    ):
        # input is the first input to the op and is typically either a Tensor or TensorList (Sequence[Tensor]).
        # This follows the typical pattern where for Tensor inputs op(t, ...) = t.op(...).
        self.input = input

        # Allow calling either as SampleInput(input, args=args, kwargs=kwargs), or as
        # SampleInput(input, *args, **kwargs) but not to mix the two forms
        if args is not None or kwargs is not None:
            assert (
                not var_args and not var_kwargs
            ), """
A SampleInput can be constructed "naturally" with *args and **kwargs or by
explicitly setting the "args" and "kwargs" parameters, but the two
methods of construction cannot be mixed!"""
        elif len(var_args) or len(var_kwargs):
            assert (
                output_process_fn_grad is None
                and broadcasts_input is None
                and name is None
            ), """
A SampleInput constructed "naturally" with *args and **kwargs
cannot specify additional metadata in keyword arguments"""

        self.args = args if args is not None else var_args
        assert isinstance(self.args, tuple)
        self.kwargs = kwargs if kwargs is not None else var_kwargs
        assert isinstance(self.kwargs, dict)

        self.output_process_fn_grad = (
            output_process_fn_grad
            if output_process_fn_grad is not None
            else lambda x: x
        )
        self.name = name if name is not None else ""

        # Specifies if `self.input` is broadcasted or not,
        # given that the operator supports broadcasting.
        # This field is used to verify the behavior for inplace variant.
        #
        # If a SampleInput is marked with `broadcasts_input=True`,
        # it is verified that we get a `RuntimeError` with this sample,
        # and inplace variant. Also inplace grad{grad} tests are skipped,
        # for such inputs (as they will error out otherwise).
        self.broadcasts_input = (
            broadcasts_input if broadcasts_input is not None else False
        )

    def with_metadata(
        self, *, output_process_fn_grad=None, broadcasts_input=None, name=None
    ):
        if output_process_fn_grad is not None:
            self.output_process_fn_grad = output_process_fn_grad
        if broadcasts_input is not None:
            self.broadcasts_input = broadcasts_input
        if name is not None:
            self.name = name
        return self

    def _repr_helper(self, formatter):
        # Helper function to return the details of the SampleInput as `str`
        # It consolidates all the fields of SampleInput and allows,
        # formatting the fields like `input`, `args`, etc with `formatter`
        # callable to customize the representation.
        # Look at `summary` method for example.
        arguments = [
            f"input={formatter(self.input)}",
            f"args={formatter(self.args)}",
            f"kwargs={formatter(self.kwargs)}",
            f"broadcasts_input={self.broadcasts_input}",
            f"name={repr(self.name)}",
        ]

        return f'SampleInput({", ".join(a for a in arguments if a is not None)})'

    def __repr__(self):
        return self._repr_helper(lambda x: x)

    def summary(self):
        # Returns the SampleInput details in a more
        # friendly format.
        # It formats `Tensor` and `TensorList`
        # in a more condensed representation.
        def formatter(arg):
            # Format any instance of `Tensor` (standalone, in list, or in dict)
            # by Tensor[TensorShape]
            # Eg. Tensor with shape (3, 4) is formatted as Tensor[3, 4]
            if isinstance(arg, torch.Tensor):
                shape = str(tuple(arg.shape))
                dtype = str(arg.dtype)
                device = str(arg.device)
                contiguity_suffix = ""
                # NB: sparse CSR tensors annoyingly return is_sparse=False
                is_sparse = arg.is_sparse or arg.layout == torch.sparse_csr
                if not is_sparse and not arg.is_contiguous():
                    contiguity_suffix = ", contiguous=False"
                return f'Tensor[size={shape}, device="{device}", dtype={dtype}{contiguity_suffix}]'
            elif isinstance(arg, dict):
                return {k: formatter(v) for k, v in arg.items()}
            elif is_iterable_of_tensors(arg):
                return "TensorList[" + ", ".join(map(formatter, arg)) + "]"
            elif isinstance(arg, (list, tuple)):  # Handle list, tuple
                return "(" + ",".join(map(formatter, arg)) + ")"

            return repr(arg)

        return self._repr_helper(formatter)

    # Applies the transform f(t) -> t to each tensor and dtype in the SampleInput
    def transform(self, f):
        def tt(t):
            def _tt(t):
                with torch.no_grad():
                    return f(t)

            if isinstance(t, torch.Tensor):
                return _tt(t)
            elif isinstance(t, torch.dtype):
                return _tt(t)
            elif isinstance(t, list):
                return list(map(tt, t))
            elif isinstance(t, tuple):
                return tuple(map(tt, t))
            elif isinstance(t, dict):
                return {k: tt(v) for k, v in t.items()}
            else:
                return t

        sample_tt_input, tt_args, tt_kwargs = (
            tt(self.input),
            tt(self.args),
            tt(self.kwargs),
        )

        # Note the transformed SampleInput assumes metadata like output_process_fn_grad is still valid!
        return SampleInput(
            sample_tt_input,
            args=tt_args,
            kwargs=tt_kwargs,
            output_process_fn_grad=self.output_process_fn_grad,
            broadcasts_input=self.broadcasts_input,
            name=self.name + "_transformed",
        )

    # Returns the NumPy version of the sample input object in the form of a tuple: (input, args, kwargs)
    # Converts tensors to ndarrays by calling .detach().cpu().numpy() on them
    # Converts dtypes by remapping them using torch_to_numpy_dtype_dict
    def numpy(self):
        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                if t.dtype is torch.bfloat16:
                    return t.detach().cpu().to(torch.float32).numpy()
                if t.dtype is torch.chalf:
                    return t.detach().cpu().to(torch.cfloat).numpy()
                return t.detach().cpu().numpy()
            elif isinstance(t, torch.dtype):
                return torch_to_numpy_dtype_dict[t]

            return t

        return self.transform(to_numpy)

    def noncontiguous(self):
        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                return noncontiguous_like(t)
            elif isinstance(t, torch.dtype):
                return t

            return t

        return self.transform(to_noncontiguous)

class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "input",
        "args",
        "kwargs",
        "output_process_fn_grad",
        "broadcasts_input",
        "name",
    ]

    def __init__(
        self,
        input,
        *var_args,
        args=None,
        kwargs=None,
        output_process_fn_grad=None,
        broadcasts_input=None,
        name=None,
        **var_kwargs,
    ):
        # input is the first input to the op and is typically either a Tensor or TensorList (Sequence[Tensor]).
        # This follows the typical pattern where for Tensor inputs op(t, ...) = t.op(...).
        self.input = input

        # Allow calling either as SampleInput(input, args=args, kwargs=kwargs), or as
        # SampleInput(input, *args, **kwargs) but not to mix the two forms
        if args is not None or kwargs is not None:
            assert (
                not var_args and not var_kwargs
            ), """
A SampleInput can be constructed "naturally" with *args and **kwargs or by
explicitly setting the "args" and "kwargs" parameters, but the two
methods of construction cannot be mixed!"""
        elif len(var_args) or len(var_kwargs):
            assert (
                output_process_fn_grad is None
                and broadcasts_input is None
                and name is None
            ), """
A SampleInput constructed "naturally" with *args and **kwargs
cannot specify additional metadata in keyword arguments"""

        self.args = args if args is not None else var_args
        assert isinstance(self.args, tuple)
        self.kwargs = kwargs if kwargs is not None else var_kwargs
        assert isinstance(self.kwargs, dict)

        self.output_process_fn_grad = (
            output_process_fn_grad
            if output_process_fn_grad is not None
            else lambda x: x
        )
        self.name = name if name is not None else ""

        # Specifies if `self.input` is broadcasted or not,
        # given that the operator supports broadcasting.
        # This field is used to verify the behavior for inplace variant.
        #
        # If a SampleInput is marked with `broadcasts_input=True`,
        # it is verified that we get a `RuntimeError` with this sample,
        # and inplace variant. Also inplace grad{grad} tests are skipped,
        # for such inputs (as they will error out otherwise).
        self.broadcasts_input = (
            broadcasts_input if broadcasts_input is not None else False
        )

    def with_metadata(
        self, *, output_process_fn_grad=None, broadcasts_input=None, name=None
    ):
        if output_process_fn_grad is not None:
            self.output_process_fn_grad = output_process_fn_grad
        if broadcasts_input is not None:
            self.broadcasts_input = broadcasts_input
        if name is not None:
            self.name = name
        return self

    def _repr_helper(self, formatter):
        # Helper function to return the details of the SampleInput as `str`
        # It consolidates all the fields of SampleInput and allows,
        # formatting the fields like `input`, `args`, etc with `formatter`
        # callable to customize the representation.
        # Look at `summary` method for example.
        arguments = [
            f"input={formatter(self.input)}",
            f"args={formatter(self.args)}",
            f"kwargs={formatter(self.kwargs)}",
            f"broadcasts_input={self.broadcasts_input}",
            f"name={repr(self.name)}",
        ]

        return f'SampleInput({", ".join(a for a in arguments if a is not None)})'

    def __repr__(self):
        return self._repr_helper(lambda x: x)

    def summary(self):
        # Returns the SampleInput details in a more
        # friendly format.
        # It formats `Tensor` and `TensorList`
        # in a more condensed representation.
        def formatter(arg):
            # Format any instance of `Tensor` (standalone, in list, or in dict)
            # by Tensor[TensorShape]
            # Eg. Tensor with shape (3, 4) is formatted as Tensor[3, 4]
            if isinstance(arg, torch.Tensor):
                shape = str(tuple(arg.shape))
                dtype = str(arg.dtype)
                device = str(arg.device)
                contiguity_suffix = ""
                # NB: sparse CSR tensors annoyingly return is_sparse=False
                is_sparse = arg.is_sparse or arg.layout == torch.sparse_csr
                if not is_sparse and not arg.is_contiguous():
                    contiguity_suffix = ", contiguous=False"
                return f'Tensor[size={shape}, device="{device}", dtype={dtype}{contiguity_suffix}]'
            elif isinstance(arg, dict):
                return {k: formatter(v) for k, v in arg.items()}
            elif is_iterable_of_tensors(arg):
                return "TensorList[" + ", ".join(map(formatter, arg)) + "]"
            elif isinstance(arg, (list, tuple)):  # Handle list, tuple
                return "(" + ",".join(map(formatter, arg)) + ")"

            return repr(arg)

        return self._repr_helper(formatter)

    # Applies the transform f(t) -> t to each tensor and dtype in the SampleInput
    def transform(self, f):
        def tt(t):
            def _tt(t):
                with torch.no_grad():
                    return f(t)

            if isinstance(t, torch.Tensor):
                return _tt(t)
            elif isinstance(t, torch.dtype):
                return _tt(t)
            elif isinstance(t, list):
                return list(map(tt, t))
            elif isinstance(t, tuple):
                return tuple(map(tt, t))
            elif isinstance(t, dict):
                return {k: tt(v) for k, v in t.items()}
            else:
                return t

        sample_tt_input, tt_args, tt_kwargs = (
            tt(self.input),
            tt(self.args),
            tt(self.kwargs),
        )

        # Note the transformed SampleInput assumes metadata like output_process_fn_grad is still valid!
        return SampleInput(
            sample_tt_input,
            args=tt_args,
            kwargs=tt_kwargs,
            output_process_fn_grad=self.output_process_fn_grad,
            broadcasts_input=self.broadcasts_input,
            name=self.name + "_transformed",
        )

    # Returns the NumPy version of the sample input object in the form of a tuple: (input, args, kwargs)
    # Converts tensors to ndarrays by calling .detach().cpu().numpy() on them
    # Converts dtypes by remapping them using torch_to_numpy_dtype_dict
    def numpy(self):
        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                if t.dtype is torch.bfloat16:
                    return t.detach().cpu().to(torch.float32).numpy()
                if t.dtype is torch.chalf:
                    return t.detach().cpu().to(torch.cfloat).numpy()
                return t.detach().cpu().numpy()
            elif isinstance(t, torch.dtype):
                return torch_to_numpy_dtype_dict[t]

            return t

        return self.transform(to_numpy)

    def noncontiguous(self):
        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                return noncontiguous_like(t)
            elif isinstance(t, torch.dtype):
                return t

            return t

        return self.transform(to_noncontiguous)

class DecorateInfo:
    """Describes which test, or type of tests, should be wrapped in the given
    decorators when testing an operator. Any test that matches all provided
    arguments will be decorated. The decorators will only be applied if the
    active_if argument is True."""

    __slots__ = [
        "decorators",
        "cls_name",
        "test_name",
        "device_type",
        "dtypes",
        "active_if",
    ]

    def __init__(
        self,
        decorators,
        cls_name=None,
        test_name=None,
        *,
        device_type=None,
        dtypes=None,
        active_if=True,
    ):
        self.decorators = (
            list(decorators)
            if isinstance(decorators, collections.abc.Sequence)
            else [decorators]
        )
        self.cls_name = cls_name
        self.test_name = test_name
        self.device_type = device_type
        self.dtypes = dtypes
        self.active_if = active_if

        # Validate dtypes
        if self.dtypes is not None:
            for dtype in self.dtypes:
                assert isinstance(dtype, torch.dtype)

    def is_active(self, cls_name, test_name, device_type, dtype, param_kwargs):
        return (
            self.active_if
            and (self.cls_name is None or self.cls_name == cls_name)
            and (self.test_name is None or self.test_name == test_name)
            and (self.device_type is None or self.device_type == device_type)
            and (self.dtypes is None or dtype in self.dtypes)
            # Support callables over kwargs to determine if the decorator is active.
            and (
                self.active_if(param_kwargs)
                if isinstance(self.active_if, Callable)
                else self.active_if
            )
        )

class DecorateInfo:
    """Describes which test, or type of tests, should be wrapped in the given
    decorators when testing an operator. Any test that matches all provided
    arguments will be decorated. The decorators will only be applied if the
    active_if argument is True."""

    __slots__ = [
        "decorators",
        "cls_name",
        "test_name",
        "device_type",
        "dtypes",
        "active_if",
    ]

    def __init__(
        self,
        decorators,
        cls_name=None,
        test_name=None,
        *,
        device_type=None,
        dtypes=None,
        active_if=True,
    ):
        self.decorators = (
            list(decorators)
            if isinstance(decorators, collections.abc.Sequence)
            else [decorators]
        )
        self.cls_name = cls_name
        self.test_name = test_name
        self.device_type = device_type
        self.dtypes = dtypes
        self.active_if = active_if

        # Validate dtypes
        if self.dtypes is not None:
            for dtype in self.dtypes:
                assert isinstance(dtype, torch.dtype)

    def is_active(self, cls_name, test_name, device_type, dtype, param_kwargs):
        return (
            self.active_if
            and (self.cls_name is None or self.cls_name == cls_name)
            and (self.test_name is None or self.test_name == test_name)
            and (self.device_type is None or self.device_type == device_type)
            and (self.dtypes is None or dtype in self.dtypes)
            # Support callables over kwargs to determine if the decorator is active.
            and (
                self.active_if(param_kwargs)
                if isinstance(self.active_if, Callable)
                else self.active_if
            )
        )

