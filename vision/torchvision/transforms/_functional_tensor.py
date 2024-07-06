def _max_value(dtype: torch.dtype) -> int:
    if dtype == torch.uint8:
        return 255
    elif dtype == torch.int8:
        return 127
    elif dtype == torch.int16:
        return 32767
    elif dtype == torch.int32:
        return 2147483647
    elif dtype == torch.int64:
        return 9223372036854775807
    else:
        # This is only here for completeness. This value is implicitly assumed in a lot of places so changing it is not
        # easy.
        return 1

def solarize(img: Tensor, threshold: float) -> Tensor:

    _assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    _assert_channels(img, [1, 3])

    if threshold > _max_value(img.dtype):
        raise TypeError("Threshold should be less than bound of img.")

    inverted_img = invert(img)
    return torch.where(img >= threshold, inverted_img, img)

