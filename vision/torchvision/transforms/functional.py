def affine(
    img: Tensor,
    angle: float,
    translate: List[int],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
    center: Optional[List[int]] = None,
) -> Tensor:
    """Apply affine transformation on the image keeping image center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): image to transform.
        angle (number): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (sequence of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or sequence): shear angle value in degrees between -180 to 180, clockwise direction.
            If a sequence is specified, the first value corresponds to a shear parallel to the x-axis, while
            the second value corresponds to a shear parallel to the y-axis.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.

    Returns:
        PIL Image or Tensor: Transformed image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(affine)

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if not isinstance(translate, (list, tuple)):
        raise TypeError("Argument translate should be a sequence")

    if len(translate) != 2:
        raise ValueError("Argument translate should be a sequence of length 2")

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    if not isinstance(shear, (numbers.Number, (list, tuple))):
        raise TypeError("Shear should be either a single value or a sequence of two values")

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError(f"Shear should be a sequence containing two values. Got {shear}")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    _, height, width = get_dimensions(img)
    if not isinstance(img, torch.Tensor):
        # center = (width * 0.5 + 0.5, height * 0.5 + 0.5)
        # it is visually better to estimate the center without 0.5 offset
        # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
        if center is None:
            center = [width * 0.5, height * 0.5]
        matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.affine(img, matrix=matrix, interpolation=pil_interpolation, fill=fill)

    center_f = [0.0, 0.0]
    if center is not None:
        _, height, width = get_dimensions(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    translate_f = [1.0 * t for t in translate]
    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)
    return F_t.affine(img, matrix=matrix, interpolation=interpolation.value, fill=fill)

def get_dimensions(img: Tensor) -> List[int]:
    """Returns the dimensions of an image as [channels, height, width].

    Args:
        img (PIL Image or Tensor): The image to be checked.

    Returns:
        List[int]: The image dimensions.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(get_dimensions)
    if isinstance(img, torch.Tensor):
        return F_t.get_dimensions(img)

    return F_pil.get_dimensions(img)

def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image. This function does not support torchscript.

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(to_pil_image)

    if isinstance(pic, torch.Tensor):
        if pic.ndim == 3:
            pic = pic.permute((1, 2, 0))
        pic = pic.numpy(force=True)
    elif not isinstance(pic, np.ndarray):
        raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")

    if pic.ndim == 2:
        # if 2D image, add channel dimension (HWC)
        pic = np.expand_dims(pic, 2)
    if pic.ndim != 3:
        raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

    if pic.shape[-1] > 4:
        raise ValueError(f"pic should not have > 4 channels. Got {pic.shape[-1]} channels.")

    npimg = pic

    if np.issubdtype(npimg.dtype, np.floating) and mode != "F":
        npimg = (npimg * 255).astype(np.uint8)

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        elif npimg.dtype == np.int16:
            expected_mode = "I;16" if sys.byteorder == "little" else "I;16B"
        elif npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(f"Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}")
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ["LA"]
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError(f"Only modes {permitted_2_channel_modes} are supported for 2D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "LA"

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK", "RGBX"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(f"Only modes {permitted_4_channel_modes} are supported for 4D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(f"Only modes {permitted_3_channel_modes} are supported for 3D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError(f"Input type {npimg.dtype} is not supported")

    return Image.fromarray(npimg, mode=mode)

def pil_to_tensor(pic: Any) -> Tensor:
    """Convert a ``PIL Image`` to a tensor of the same type.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.PILToTensor` for more details.

    .. note::

        A deep copy of the underlying array is performed.

    Args:
        pic (PIL Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(pil_to_tensor)
    if not F_pil._is_pil_image(pic):
        raise TypeError(f"pic should be PIL Image. Got {type(pic)}")

    if accimage is not None and isinstance(pic, accimage.Image):
        # accimage format is always uint8 internally, so always return uint8 here
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.uint8)
        pic.copyto(nppic)
        return torch.as_tensor(nppic)

    # handle PIL Image
    img = torch.as_tensor(np.array(pic, copy=True))
    img = img.view(pic.size[1], pic.size[0], F_pil.get_image_num_channels(pic))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1))
    return img

def resize(
    img: Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = True,
) -> Tensor:
    r"""Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.NEAREST_EXACT``, ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image. If the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``,
            ``size`` will be overruled so that the longer edge is equal to
            ``max_size``.
            As a result, the smaller edge may be shorter than ``size``. This
            is only supported if ``size`` is an int (or a sequence of length
            1 in torchscript mode).
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True`` (default): will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The default value changed from ``None`` to ``True`` in
            v0.17, for the PIL and Tensor backends to be consistent.

    Returns:
        PIL Image or Tensor: Resized image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(resize)

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if isinstance(size, (list, tuple)):
        if len(size) not in [1, 2]:
            raise ValueError(
                f"Size must be an int or a 1 or 2 element tuple/list, not a {len(size)} element tuple/list"
            )
        if max_size is not None and len(size) != 1:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller edge, "
                "i.e. size should be an int or a sequence of length 1 in torchscript mode."
            )

    _, image_height, image_width = get_dimensions(img)
    if isinstance(size, int):
        size = [size]
    output_size = _compute_resized_output_size((image_height, image_width), size, max_size)

    if [image_height, image_width] == output_size:
        return img

    if not isinstance(img, torch.Tensor):
        if antialias is False:
            warnings.warn("Anti-alias option is always applied for PIL Image input. Argument antialias is ignored.")
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.resize(img, size=output_size, interpolation=pil_interpolation)

    return F_t.resize(img, size=output_size, interpolation=interpolation.value, antialias=antialias)

def perspective(
    img: Tensor,
    startpoints: List[List[int]],
    endpoints: List[List[int]],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None,
) -> Tensor:
    """Perform perspective transform of the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): Image to be transformed.
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.

    Returns:
        PIL Image or Tensor: transformed Image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(perspective)

    coeffs = _get_perspective_coeffs(startpoints, endpoints)

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if not isinstance(img, torch.Tensor):
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.perspective(img, coeffs, interpolation=pil_interpolation, fill=fill)

    return F_t.perspective(img, coeffs, interpolation=interpolation.value, fill=fill)

def normalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(normalize)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"img should be Tensor Image. Got {type(tensor)}")

    return F_t.normalize(tensor, mean=mean, std=std, inplace=inplace)

def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image. This function does not support torchscript.

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(to_pil_image)

    if isinstance(pic, torch.Tensor):
        if pic.ndim == 3:
            pic = pic.permute((1, 2, 0))
        pic = pic.numpy(force=True)
    elif not isinstance(pic, np.ndarray):
        raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")

    if pic.ndim == 2:
        # if 2D image, add channel dimension (HWC)
        pic = np.expand_dims(pic, 2)
    if pic.ndim != 3:
        raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

    if pic.shape[-1] > 4:
        raise ValueError(f"pic should not have > 4 channels. Got {pic.shape[-1]} channels.")

    npimg = pic

    if np.issubdtype(npimg.dtype, np.floating) and mode != "F":
        npimg = (npimg * 255).astype(np.uint8)

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        elif npimg.dtype == np.int16:
            expected_mode = "I;16" if sys.byteorder == "little" else "I;16B"
        elif npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(f"Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}")
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ["LA"]
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError(f"Only modes {permitted_2_channel_modes} are supported for 2D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "LA"

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK", "RGBX"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(f"Only modes {permitted_4_channel_modes} are supported for 4D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(f"Only modes {permitted_3_channel_modes} are supported for 3D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError(f"Input type {npimg.dtype} is not supported")

    return Image.fromarray(npimg, mode=mode)

def rotate(
    img: Tensor,
    angle: float,
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    expand: bool = False,
    center: Optional[List[int]] = None,
    fill: Optional[List[float]] = None,
) -> Tensor:
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): image to be rotated.
        angle (number): rotation angle value in degrees, counter-clockwise.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.
    Returns:
        PIL Image or Tensor: Rotated image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(rotate)

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    if not isinstance(img, torch.Tensor):
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.rotate(img, angle=angle, interpolation=pil_interpolation, expand=expand, center=center, fill=fill)

    center_f = [0.0, 0.0]
    if center is not None:
        _, height, width = get_dimensions(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    return F_t.rotate(img, matrix=matrix, interpolation=interpolation.value, expand=expand, fill=fill)

def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    """Convert a tensor image to the given ``dtype`` and scale the values accordingly
    This function does not support PIL Image.

    Args:
        image (torch.Tensor): Image to be converted
        dtype (torch.dtype): Desired data type of the output

    Returns:
        Tensor: Converted image

    .. note::

        When converting from a smaller to a larger integer ``dtype`` the maximum values are **not** mapped exactly.
        If converted back and forth, this mismatch has no effect.

    Raises:
        RuntimeError: When trying to cast :class:`torch.float32` to :class:`torch.int32` or :class:`torch.int64` as
            well as for trying to cast :class:`torch.float64` to :class:`torch.int64`. These conversions might lead to
            overflow errors since the floating point ``dtype`` cannot store consecutive integers over the whole range
            of the integer ``dtype``.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(convert_image_dtype)
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input img should be Tensor Image")

    return F_t.convert_image_dtype(image, dtype)

def elastic_transform(
    img: Tensor,
    displacement: Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None,
) -> Tensor:
    """Transform a tensor image with elastic transformations.
    Given alpha and sigma, it will generate displacement
    vectors for all pixels based on random offsets. Alpha controls the strength
    and sigma controls the smoothness of the displacements.
    The displacements are added to an identity grid and the resulting grid is
    used to grid_sample from the image.

    Applications:
        Randomly transforms the morphology of objects in images and produces a
        see-through-water-like effect.

    Args:
        img (PIL Image or Tensor): Image on which elastic_transform is applied.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".
        displacement (Tensor): The displacement field. Expected shape is [1, H, W, 2].
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0.
            If a tuple of length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(elastic_transform)
    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum."
        )
        interpolation = _interpolation_modes_from_int(interpolation)

    if not isinstance(displacement, torch.Tensor):
        raise TypeError("Argument displacement should be a Tensor")

    t_img = img
    if not isinstance(img, torch.Tensor):
        if not F_pil._is_pil_image(img):
            raise TypeError(f"img should be PIL Image or Tensor. Got {type(img)}")
        t_img = pil_to_tensor(img)

    shape = t_img.shape
    shape = (1,) + shape[-2:] + (2,)
    if shape != displacement.shape:
        raise ValueError(f"Argument displacement shape should be {shape}, but given {displacement.shape}")

    # TODO: if image shape is [N1, N2, ..., C, H, W] and
    # displacement is [1, H, W, 2] we need to reshape input image
    # such grid_sampler takes internal code for 4D input

    output = F_t.elastic_transform(
        t_img,
        displacement,
        interpolation=interpolation.value,
        fill=fill,
    )

    if not isinstance(img, torch.Tensor):
        output = to_pil_image(output, mode=img.mode)
    return output

def perspective(
    img: Tensor,
    startpoints: List[List[int]],
    endpoints: List[List[int]],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None,
) -> Tensor:
    """Perform perspective transform of the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): Image to be transformed.
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.

    Returns:
        PIL Image or Tensor: transformed Image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(perspective)

    coeffs = _get_perspective_coeffs(startpoints, endpoints)

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if not isinstance(img, torch.Tensor):
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.perspective(img, coeffs, interpolation=pil_interpolation, fill=fill)

    return F_t.perspective(img, coeffs, interpolation=interpolation.value, fill=fill)

