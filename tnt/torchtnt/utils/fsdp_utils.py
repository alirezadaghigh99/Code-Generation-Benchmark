class MixedPrecision:
    """Supported values for `MixedPrecision <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision>`_"""

    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    buffer_dtype: Optional[str] = None
    keep_low_precision_grads: bool = False
    cast_forward_inputs: bool = False
    cast_root_forward_inputs: bool = True
    _module_classes_to_ignore: Sequence[str] = (
        "torch.nn.modules.batchnorm._BatchNorm",
    )

    def to_native_mixed_precision(self) -> _MixedPrecision:
        """Convert this instance to its PyTorch native MixedPrecision."""

        # Convert string module classes to their corresponding types
        # e.g. "torch.nn.modules.batchnorm._BatchNorm" -> torch.nn.modules.batchnorm._BatchNorm
        target_types: List[Type[torch.nn.Module]] = []
        for type_str in self._module_classes_to_ignore:
            path, _, attr = type_str.rpartition(".")
            try:
                target_types.append(getattr(importlib.import_module(path), attr))
            except (AttributeError, ModuleNotFoundError) as e:
                raise ValueError(f"Invalid module class '{type_str}': {e}")
        module_classes_to_ignore: Sequence[Type[torch.nn.Module]] = target_types

        return _MixedPrecision(
            param_dtype=_to_dtype_or_none(self.param_dtype),
            reduce_dtype=_to_dtype_or_none(self.reduce_dtype),
            buffer_dtype=_to_dtype_or_none(self.buffer_dtype),
            keep_low_precision_grads=self.keep_low_precision_grads,
            cast_forward_inputs=self.cast_forward_inputs,
            cast_root_forward_inputs=self.cast_root_forward_inputs,
            _module_classes_to_ignore=module_classes_to_ignore,
        )

