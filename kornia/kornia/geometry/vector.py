def from_coords(
        cls,
        x: Union[float, Tensor],
        y: Union[float, Tensor],
        z: Union[float, Tensor],
        device: Optional[Device] = None,
        dtype: Dtype = None,
    ) -> "Vector3":
        KORNIA_CHECK(type(x) == type(y) == type(z))
        KORNIA_CHECK(isinstance(x, (Tensor, float)))
        if isinstance(x, float):
            return wrap(as_tensor((x, y, z), device=device, dtype=dtype), Vector3)
        # TODO: this is totally insane ...
        tensors: Tuple[Tensor, ...] = (x, cast(Tensor, y), cast(Tensor, z))
        return wrap(stack(tensors, -1), Vector3)

