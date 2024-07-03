def jpeg_codec_differentiable(
    image_rgb: Tensor,
    jpeg_quality: Tensor,
    quantization_table_y: Tensor | None = None,
    quantization_table_c: Tensor | None = None,
) -> Tensor:
    r"""Differentiable JPEG encoding-decoding module.

    Based on :cite:`reich2024` :cite:`shin2017`, we perform differentiable JPEG encoding-decoding as follows:

    .. image:: _static/img/jpeg_codec_differentiable.png

    .. math::

        \text{JPEG}_{\text{diff}}(I, q, QT_{y}, QT_{c}) = \hat{I}

    Where:
       - :math:`I` is the original image to be coded.
       - :math:`q` is the JPEG quality controlling the compression strength.
       - :math:`QT_{y}` is the luma quantization table.
       - :math:`QT_{c}` is the chroma quantization table.
       - :math:`\hat{I}` is the resulting JPEG encoded-decoded image.

    .. note:::
        The input (and output) pixel range is :math:`[0, 1]`. In case you want to handle normalized images you are
        required to first perform denormalization followed by normalizing the output images again.

        Note, that this implementation models the encoding-decoding mapping of JPEG in a differentiable setting,
        however, does not allow the excess of the JPEG-coded byte file itself.
        For more details please refer to :cite:`reich2024`.

        This implementation is not meant for data loading. For loading JPEG images please refer to `kornia.io`.
        There we provide an optimized Rust implementation for fast JPEG loading.

    Args:
        image_rgb: the RGB image to be coded.
        jpeg_quality: JPEG quality in the range :math:`[0, 100]` controlling the compression strength.
        quantization_table_y: quantization table for Y channel. Default: `None`, which will load the standard
          quantization table.
        quantization_table_c: quantization table for C channels. Default: `None`, which will load the standard
          quantization table.

    Shape:
        - image_rgb: :math:`(*, 3, H, W)`.
        - jpeg_quality: :math:`(1)` or :math:`(B)` (if used batch dim. needs to match w/ image_rgb).
        - quantization_table_y: :math:`(8, 8)` or :math:`(B, 8, 8)` (if used batch dim. needs to match w/ image_rgb).
        - quantization_table_c: :math:`(8, 8)` or :math:`(B, 8, 8)` (if used batch dim. needs to match w/ image_rgb).

    Return:
        JPEG coded image of the shape :math:`(B, 3, H, W)`

    Example:

        To perform JPEG coding with the standard quantization tables just provide a JPEG quality

        >>> img = torch.rand(3, 3, 64, 64, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.tensor((99.0, 25.0, 1.0), requires_grad=True)
        >>> img_jpeg = jpeg_codec_differentiable(img, jpeg_quality)
        >>> img_jpeg.sum().backward()

        You also have the option to provide custom quantization tables

        >>> img = torch.rand(3, 3, 64, 64, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.tensor((99.0, 25.0, 1.0), requires_grad=True)
        >>> quantization_table_y = torch.randint(1, 256, size=(3, 8, 8), dtype=torch.float)
        >>> quantization_table_c = torch.randint(1, 256, size=(3, 8, 8), dtype=torch.float)
        >>> img_jpeg = jpeg_codec_differentiable(img, jpeg_quality, quantization_table_y, quantization_table_c)
        >>> img_jpeg.sum().backward()

        In case you want to control the quantization purly base on the quantization tables use a JPEG quality of 99.5.
        Setting the JPEG quality to 99.5 leads to a QT scaling of 1, see Eq. 2 of :cite:`reich2024` for details.

        >>> img = torch.rand(3, 3, 64, 64, requires_grad=True, dtype=torch.float)
        >>> jpeg_quality = torch.ones(3) * 99.5
        >>> quantization_table_y = torch.randint(1, 256, size=(3, 8, 8), dtype=torch.float)
        >>> quantization_table_c = torch.randint(1, 256, size=(3, 8, 8), dtype=torch.float)
        >>> img_jpeg = jpeg_codec_differentiable(img, jpeg_quality, quantization_table_y, quantization_table_c)
        >>> img_jpeg.sum().backward()
    """
    # Check that inputs are tensors
    KORNIA_CHECK_IS_TENSOR(image_rgb)
    KORNIA_CHECK_IS_TENSOR(jpeg_quality)
    # Get device and dtype
    dtype: Dtype = image_rgb.dtype
    device: Device = image_rgb.device
    # Use default QT if QT is not given
    quantization_table_y = _get_default_qt_y(device, dtype) if quantization_table_y is None else quantization_table_y
    quantization_table_c = _get_default_qt_c(device, dtype) if quantization_table_c is None else quantization_table_c
    KORNIA_CHECK_IS_TENSOR(quantization_table_y)
    KORNIA_CHECK_IS_TENSOR(quantization_table_c)
    # Check shape of inputs
    KORNIA_CHECK_SHAPE(image_rgb, ["*", "3", "H", "W"])
    KORNIA_CHECK_SHAPE(jpeg_quality, ["B"])
    # Add batch dimension to quantization tables if needed
    if quantization_table_y.ndim == 2:
        quantization_table_y = quantization_table_y.unsqueeze(dim=0)
    if quantization_table_c.ndim == 2:
        quantization_table_c = quantization_table_c.unsqueeze(dim=0)
    # Check resulting shape of quantization tables
    KORNIA_CHECK_SHAPE(quantization_table_y, ["B", "8", "8"])
    KORNIA_CHECK_SHAPE(quantization_table_c, ["B", "8", "8"])
    # Check value range of JPEG quality
    KORNIA_CHECK(
        (jpeg_quality.amin().item() >= 0.0) and (jpeg_quality.amax().item() <= 100.0),
        f"JPEG quality is out of range. Expected range is [0, 100], "
        f"got [{jpeg_quality.amin().item()}, {jpeg_quality.amax().item()}]. Consider clipping jpeg_quality.",
    )
    # Pad the image to a shape dividable by 16
    image_rgb, h_pad, w_pad = _perform_padding(image_rgb)
    # Get height and shape
    H, W = image_rgb.shape[-2:]
    # Check matching batch dimensions
    if quantization_table_y.shape[0] != 1:
        KORNIA_CHECK(
            quantization_table_y.shape[0] == image_rgb.shape[0],
            f"Batch dimensions do not match. "
            f"Got {image_rgb.shape[0]} images and {quantization_table_y.shape[0]} quantization tables (Y).",
        )
    if quantization_table_c.shape[0] != 1:
        KORNIA_CHECK(
            quantization_table_c.shape[0] == image_rgb.shape[0],
            f"Batch dimensions do not match. "
            f"Got {image_rgb.shape[0]} images and {quantization_table_c.shape[0]} quantization tables (C).",
        )
    if jpeg_quality.shape[0] != 1:
        KORNIA_CHECK(
            jpeg_quality.shape[0] == image_rgb.shape[0],
            f"Batch dimensions do not match. "
            f"Got {image_rgb.shape[0]} images and {jpeg_quality.shape[0]} JPEG qualities.",
        )
    # Quantization tables to same device and dtype as input image
    quantization_table_y = quantization_table_y.to(device, dtype)
    quantization_table_c = quantization_table_c.to(device, dtype)
    # Perform encoding
    y_encoded, cb_encoded, cr_encoded = _jpeg_encode(
        image_rgb=image_rgb,
        jpeg_quality=jpeg_quality,
        quantization_table_c=quantization_table_c,
        quantization_table_y=quantization_table_y,
    )
    image_rgb_jpeg: Tensor = _jpeg_decode(
        input_y=y_encoded,
        input_cb=cb_encoded,
        input_cr=cr_encoded,
        jpeg_quality=jpeg_quality,
        H=H,
        W=W,
        quantization_table_c=quantization_table_c,
        quantization_table_y=quantization_table_y,
    )
    # Clip coded image
    image_rgb_jpeg = differentiable_clipping(input=image_rgb_jpeg, min_val=0.0, max_val=255.0)
    # Crop the image again to the original shape
    image_rgb_jpeg = image_rgb_jpeg[..., : H - h_pad, : W - w_pad]
    return image_rgb_jpeg