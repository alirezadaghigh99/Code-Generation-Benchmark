def brightness_contrast_adjust(
    img: np.ndarray,
    alpha: float = 1,
    beta: float = 0,
    beta_by_max: bool = False,
) -> np.ndarray:
    if beta_by_max:
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        value = beta * max_value
    else:
        value = beta * np.mean(img)

    return multiply_add(img, alpha, value)def equalize(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    mode: ImageMode = "cv",
    by_channels: bool = True,
) -> np.ndarray:
    _check_preconditions(img, mask, by_channels)

    function = _equalize_pil if mode == "pil" else _equalize_cv

    if is_grayscale_image(img):
        return function(img, _handle_mask(mask))

    if not by_channels:
        result_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        result_img[..., 0] = function(result_img[..., 0], _handle_mask(mask))
        return cv2.cvtColor(result_img, cv2.COLOR_YCrCb2RGB)

    result_img = np.empty_like(img)
    for i in range(3):
        _mask = _handle_mask(mask, i)
        result_img[..., i] = function(img[..., i], _mask)

    return result_imgdef from_float(img: np.ndarray, dtype: np.dtype, max_value: float | None = None) -> np.ndarray:
    if max_value is None:
        if dtype not in MAX_VALUES_BY_DTYPE:
            msg = (
                f"Can't infer the maximum value for dtype {dtype}. "
                "You need to specify the maximum value manually by passing the max_value argument."
            )
            raise RuntimeError(msg)
        max_value = MAX_VALUES_BY_DTYPE[dtype]
    return (img * max_value).astype(dtype)def gamma_transform(img: np.ndarray, gamma: float) -> np.ndarray:
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        return cv2.LUT(img, table.astype(np.uint8))
    return np.power(img, gamma)def downscale(
    img: np.ndarray,
    scale: float,
    down_interpolation: int = cv2.INTER_AREA,
    up_interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    height, width = img.shape[:2]

    need_cast = (
        up_interpolation != cv2.INTER_NEAREST or down_interpolation != cv2.INTER_NEAREST
    ) and img.dtype == np.uint8
    if need_cast:
        img = to_float(img)
    downscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=down_interpolation)
    upscaled = cv2.resize(downscaled, (width, height), interpolation=up_interpolation)
    if need_cast:
        return from_float(np.clip(upscaled, 0, 1), dtype=np.dtype("uint8"))
    return upscaled