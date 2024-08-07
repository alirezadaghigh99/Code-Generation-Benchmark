class _BasicAugmentationBase(Module):
    r"""_BasicAugmentationBase base class for customized augmentation implementations.

    Plain augmentation base class without the functionality of transformation matrix calculations.
    By default, the random computations will be happened on CPU with ``torch.get_default_dtype()``.
    To change this behaviour, please use ``set_rng_device_and_dtype``.

    For automatically generating the corresponding ``__repr__`` with full customized parameters, you may need to
    implement ``_param_generator`` by inheriting ``RandomGeneratorBase`` for generating random parameters and
    put all static parameters inside ``self.flags``. You may take the advantage of ``PlainUniformGenerator`` to
    generate simple uniform parameters with less boilerplate code.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities element-wise.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to
          the batch form ``False``.
    """

    def __init__(
        self,
        p: float = 0.5,
        p_batch: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__()
        self.p = p
        self.p_batch = p_batch
        self.same_on_batch = same_on_batch
        self.keepdim = keepdim
        self._params: Dict[str, Tensor] = {}
        self._p_gen: Distribution
        self._p_batch_gen: Distribution
        if p != 0.0 or p != 1.0:
            self._p_gen = Bernoulli(self.p)
        if p_batch != 0.0 or p_batch != 1.0:
            self._p_batch_gen = Bernoulli(self.p_batch)
        self._param_generator: Optional[RandomGeneratorBase] = None
        self.flags: Dict[str, Any] = {}
        self.set_rng_device_and_dtype(torch.device("cpu"), torch.get_default_dtype())

    apply_transform: Callable[..., Tensor] = _apply_transform_unimplemented

    def to(self, *args: Any, **kwargs: Any) -> "_BasicAugmentationBase":
        r"""Set the device and dtype for the random number generator."""
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.set_rng_device_and_dtype(device, dtype)
        return super().to(*args, **kwargs)

    def __repr__(self) -> str:
        txt = f"p={self.p}, p_batch={self.p_batch}, same_on_batch={self.same_on_batch}"
        if isinstance(self._param_generator, RandomGeneratorBase):
            txt = f"{self._param_generator!s}, {txt}"
        for k, v in self.flags.items():
            if isinstance(v, Enum):
                txt += f", {k}={v.name.lower()}"
            else:
                txt += f", {k}={v}"
        return f"{self.__class__.__name__}({txt})"

    def __unpack_input__(self, input: Tensor) -> Tensor:
        return input

    def transform_tensor(
        self,
        input: Tensor,
        *,
        shape: Optional[Tensor] = None,
        match_channel: bool = True,
    ) -> Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def validate_tensor(self, input: Tensor) -> None:
        """Check if the input tensor is formatted as expected."""
        raise NotImplementedError

    def transform_output_tensor(self, output: Tensor, output_shape: Tuple[int, ...]) -> Tensor:
        """Standardize output tensors."""
        return _transform_output_shape(output, output_shape) if self.keepdim else output

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        if self._param_generator is not None:
            return self._param_generator(batch_shape, self.same_on_batch)
        return {}

    def set_rng_device_and_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        """Change the random generation device and dtype.

        Note:
            The generated random numbers are not reproducible across different devices and dtypes.
        """
        self.device = device
        self.dtype = dtype
        if self._param_generator is not None:
            self._param_generator.set_rng_device_and_dtype(device, dtype)

    def __batch_prob_generator__(
        self,
        batch_shape: Tuple[int, ...],
        p: float,
        p_batch: float,
        same_on_batch: bool,
    ) -> Tensor:
        batch_prob: Tensor
        if p_batch == 1:
            batch_prob = zeros(1) + 1
        elif p_batch == 0:
            batch_prob = zeros(1)
        elif isinstance(self._p_batch_gen, (RelaxedBernoulli,)):
            # NOTE: there is no simple way to know if the sampler has `rsample` or not
            batch_prob = _adapted_rsampling((1,), self._p_batch_gen, same_on_batch)
        else:
            batch_prob = _adapted_sampling((1,), self._p_batch_gen, same_on_batch)

        if batch_prob.sum() == 1:
            elem_prob: Tensor
            if p == 1:
                elem_prob = zeros(batch_shape[0]) + 1
            elif p == 0:
                elem_prob = zeros(batch_shape[0])
            elif isinstance(self._p_gen, (RelaxedBernoulli,)):
                elem_prob = _adapted_rsampling((batch_shape[0],), self._p_gen, same_on_batch)
            else:
                elem_prob = _adapted_sampling((batch_shape[0],), self._p_gen, same_on_batch)
            batch_prob = batch_prob * elem_prob
        else:
            batch_prob = batch_prob.repeat(batch_shape[0])
        if len(batch_prob.shape) == 2:
            return batch_prob[..., 0]
        return batch_prob

    def _process_kwargs_to_params_and_flags(
        self,
        params: Optional[Dict[str, Tensor]] = None,
        flags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        # NOTE: determine how to save self._params
        save_kwargs = kwargs["save_kwargs"] if "save_kwargs" in kwargs else False

        params = self._params if params is None else params
        flags = self.flags if flags is None else flags

        if save_kwargs:
            params = override_parameters(params, kwargs, in_place=True)
            self._params = params
        else:
            self._params = params
            params = override_parameters(params, kwargs, in_place=False)

        flags = override_parameters(flags, kwargs, in_place=False)
        return params, flags

    def forward_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        batch_prob = self.__batch_prob_generator__(batch_shape, self.p, self.p_batch, self.same_on_batch)
        to_apply = batch_prob > 0.5
        _params = self.generate_parameters(torch.Size((int(to_apply.sum().item()), *batch_shape[1:])))
        if _params is None:
            _params = {}
        _params["batch_prob"] = batch_prob
        # Added another input_size parameter for geometric transformations
        # This might be needed for correctly inversing.
        input_size = tensor(batch_shape, dtype=torch.long)
        _params.update({"forward_input_shape": input_size})
        return _params

    def apply_func(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.apply_transform(input, params, flags)

    def forward(self, input: Tensor, params: Optional[Dict[str, Tensor]] = None, **kwargs: Any) -> Tensor:
        """Perform forward operations.

        Args:
            input: the input tensor.
            params: the corresponding parameters for an operation.
                If None, a new parameter suite will be generated.
            **kwargs: key-value pairs to override the parameters and flags.

        Note:
            By default, all the overwriting parameters in kwargs will not be recorded
            as in ``self._params``. If you wish it to be recorded, you may pass
            ``save_kwargs=True`` additionally.
        """
        in_tensor = self.__unpack_input__(input)
        input_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if params is None:
            params = self.forward_parameters(batch_shape)

        if "batch_prob" not in params:
            params["batch_prob"] = tensor([True] * batch_shape[0])

        params, flags = self._process_kwargs_to_params_and_flags(params, self.flags, **kwargs)

        output = self.apply_func(in_tensor, params, flags)
        return self.transform_output_tensor(output, input_shape) if self.keepdim else output

