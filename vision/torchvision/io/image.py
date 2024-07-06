def read_image(
    path: str,
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
    apply_exif_orientation: bool = False,
) -> torch.Tensor:
    """
    Reads a JPEG, PNG or GIF image into a 3 dimensional RGB or grayscale Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 in [0, 255].

    Args:
        path (str or ``pathlib.Path``): path of the JPEG, PNG or GIF image.
        mode (ImageReadMode): the read mode used for optionally converting the image.
            Default: ``ImageReadMode.UNCHANGED``.
            See ``ImageReadMode`` class for more information on various
            available modes. Ignored for GIFs.
        apply_exif_orientation (bool): apply EXIF orientation transformation to the output tensor.
            Ignored for GIFs. Default: False.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(read_image)
    data = read_file(path)
    return decode_image(data, mode, apply_exif_orientation=apply_exif_orientation)

def read_file(path: str) -> torch.Tensor:
    """
    Reads and outputs the bytes contents of a file as a uint8 Tensor
    with one dimension.

    Args:
        path (str or ``pathlib.Path``): the path to the file to be read

    Returns:
        data (Tensor)
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(read_file)
    data = torch.ops.image.read_file(str(path))
    return data

def encode_jpeg(
    input: Union[torch.Tensor, List[torch.Tensor]], quality: int = 75
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Takes a (list of) input tensor(s) in CHW layout and returns a (list of) buffer(s) with the contents
    of the corresponding JPEG file(s).

    .. note::
        Passing a list of CUDA tensors is more efficient than repeated individual calls to ``encode_jpeg``.
        For CPU tensors the performance is equivalent.

    Args:
        input (Tensor[channels, image_height, image_width] or List[Tensor[channels, image_height, image_width]]):
            (list of) uint8 image tensor(s) of ``c`` channels, where ``c`` must be 1 or 3
        quality (int): Quality of the resulting JPEG file(s). Must be a number between
            1 and 100. Default: 75

    Returns:
        output (Tensor[1] or list[Tensor[1]]): A (list of) one dimensional uint8 tensor(s) that contain the raw bytes of the JPEG file.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(encode_jpeg)
    if quality < 1 or quality > 100:
        raise ValueError("Image quality should be a positive number between 1 and 100")
    if isinstance(input, list):
        if not input:
            raise ValueError("encode_jpeg requires at least one input tensor when a list is passed")
        if input[0].device.type == "cuda":
            return torch.ops.image.encode_jpegs_cuda(input, quality)
        else:
            return [torch.ops.image.encode_jpeg(image, quality) for image in input]
    else:  # single input tensor
        if input.device.type == "cuda":
            return torch.ops.image.encode_jpegs_cuda([input], quality)[0]
        else:
            return torch.ops.image.encode_jpeg(input, quality)

def decode_jpeg(
    input: torch.Tensor,
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
    device: str = "cpu",
    apply_exif_orientation: bool = False,
) -> torch.Tensor:
    """
    Decodes a JPEG image into a 3 dimensional RGB or grayscale Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 between 0 and 255.

    Args:
        input (Tensor[1]): a one dimensional uint8 tensor containing
            the raw bytes of the JPEG image. This tensor must be on CPU,
            regardless of the ``device`` parameter.
        mode (ImageReadMode): the read mode used for optionally
            converting the image. The supported modes are: ``ImageReadMode.UNCHANGED``,
            ``ImageReadMode.GRAY`` and ``ImageReadMode.RGB``
            Default: ``ImageReadMode.UNCHANGED``.
            See ``ImageReadMode`` class for more information on various
            available modes.
        device (str or torch.device): The device on which the decoded image will
            be stored. If a cuda device is specified, the image will be decoded
            with `nvjpeg <https://developer.nvidia.com/nvjpeg>`_. This is only
            supported for CUDA version >= 10.1

            .. betastatus:: device parameter

            .. warning::
                There is a memory leak in the nvjpeg library for CUDA versions < 11.6.
                Make sure to rely on CUDA 11.6 or above before using ``device="cuda"``.
        apply_exif_orientation (bool): apply EXIF orientation transformation to the output tensor.
            Default: False. Only implemented for JPEG format on CPU.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(decode_jpeg)
    device = torch.device(device)
    if device.type == "cuda":
        output = torch.ops.image.decode_jpeg_cuda(input, mode.value, device)
    else:
        output = torch.ops.image.decode_jpeg(input, mode.value, apply_exif_orientation)
    return output

def decode_image(
    input: torch.Tensor,
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
    apply_exif_orientation: bool = False,
) -> torch.Tensor:
    """
    Detect whether an image is a JPEG, PNG or GIF and performs the appropriate
    operation to decode the image into a 3 dimensional RGB or grayscale Tensor.

    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 in [0, 255].

    Args:
        input (Tensor): a one dimensional uint8 tensor containing the raw bytes of the
            PNG or JPEG image.
        mode (ImageReadMode): the read mode used for optionally converting the image.
            Default: ``ImageReadMode.UNCHANGED``.
            See ``ImageReadMode`` class for more information on various
            available modes. Ignored for GIFs.
        apply_exif_orientation (bool): apply EXIF orientation transformation to the output tensor.
            Ignored for GIFs. Default: False.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(decode_image)
    output = torch.ops.image.decode_image(input, mode.value, apply_exif_orientation)
    return output

