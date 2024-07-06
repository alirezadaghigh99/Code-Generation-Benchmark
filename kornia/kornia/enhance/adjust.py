def invert(image: Tensor, max_val: Tensor = Tensor([1.0])) -> Tensor:
    r"""Invert the values of an input image tensor by its maximum value.

    .. image:: _static/img/invert.png

    Args:
        image: The input tensor to invert with an arbitatry shape.
        max_val: The expected maximum value in the input tensor. The shape has to
          according to the input tensor shape, or at least has to work with broadcasting.

    Example:
        >>> img = torch.rand(1, 2, 4, 4)
        >>> invert(img).shape
        torch.Size([1, 2, 4, 4])

        >>> img = 255. * torch.rand(1, 2, 3, 4, 4)
        >>> invert(img, torch.as_tensor(255.)).shape
        torch.Size([1, 2, 3, 4, 4])

        >>> img = torch.rand(1, 3, 4, 4)
        >>> invert(img, torch.as_tensor([[[[1.]]]])).shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(image, Tensor):
        raise AssertionError(f"Input is not a Tensor. Got: {type(input)}")

    if not isinstance(max_val, Tensor):
        raise AssertionError(f"max_val is not a Tensor. Got: {type(max_val)}")

    return max_val.to(image) - image

