def to_image(inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> tv_tensors.Image:
    """See :class:`~torchvision.transforms.v2.ToImage` for details."""
    if isinstance(inpt, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(inpt)).permute((2, 0, 1)).contiguous()
    elif isinstance(inpt, PIL.Image.Image):
        output = pil_to_tensor(inpt)
    elif isinstance(inpt, torch.Tensor):
        output = inpt
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, or a PIL image, but got {type(inpt)} instead."
        )
    return tv_tensors.Image(output)

