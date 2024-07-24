class TransposeDimensions(Transform):
    _transformed_types = (is_pure_tensor, tv_tensors.Image, tv_tensors.Video)

    def __init__(self, dims: Union[Tuple[int, int], Dict[Type, Optional[Tuple[int, int]]]]) -> None:
        super().__init__()
        if not isinstance(dims, dict):
            dims = _get_defaultdict(dims)
        if torch.Tensor in dims and any(cls in dims for cls in [tv_tensors.Image, tv_tensors.Video]):
            warnings.warn(
                "Got `dims` values for `torch.Tensor` and either `tv_tensors.Image` or `tv_tensors.Video`. "
                "Note that a plain `torch.Tensor` will *not* be transformed by this (or any other transformation) "
                "in case a `tv_tensors.Image` or `tv_tensors.Video` is present in the input."
            )
        self.dims = dims

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> torch.Tensor:
        dims = self.dims[type(inpt)]
        if dims is None:
            return inpt.as_subclass(torch.Tensor)
        return inpt.transpose(*dims)

