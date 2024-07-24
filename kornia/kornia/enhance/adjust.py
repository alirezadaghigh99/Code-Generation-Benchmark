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

class AdjustGamma(Module):
    r"""Perform gamma correction on an image.

    The input image is expected to be in the range of [0, 1].

    Args:
        gamma: Non negative real number, same as y\gammay in the equation.
          gamma larger than 1 make the shadows darker, while gamma smaller than 1 make
          dark regions lighter.
        gain: The constant multiplier.

    Shape:
        - Input: Image to be adjusted in the shape of :math:`(*, N)`.
        - Output: Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustGamma(1.0, 2.0)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y1 = torch.ones(2) * 1.0
        >>> y2 = torch.ones(2) * 2.0
        >>> AdjustGamma(y1, y2)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    def __init__(self, gamma: Union[float, Tensor], gain: Union[float, Tensor] = 1.0) -> None:
        super().__init__()
        self.gamma: Union[float, Tensor] = gamma
        self.gain: Union[float, Tensor] = gain

    def forward(self, input: Tensor) -> Tensor:
        return adjust_gamma(input, self.gamma, self.gain)

class AdjustContrastWithMeanSubtraction(Module):
    r"""Adjust Contrast of an image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.
    The input image is expected to be in the range of [0, 1].

    Args:
        contrast_factor: Contrast adjust factor per element
          in the batch by subtracting its mean grayscaled version.
          0 generates a completely black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Shape:
        - Input: Image/Input to be adjusted in the shape of :math:`(*, N)`.
        - Output: Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustContrastWithMeanSubtraction(0.5)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.ones(2)
        >>> AdjustContrastWithMeanSubtraction(y)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    def __init__(self, contrast_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.contrast_factor: Union[float, Tensor] = contrast_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_contrast_with_mean_subtraction(input, self.contrast_factor)

class AdjustContrast(Module):
    r"""Adjust Contrast of an image.

    This implementation aligns OpenCV, not PIL. Hence, the output differs from TorchVision.
    The input image is expected to be in the range of [0, 1].

    Args:
        contrast_factor: Contrast adjust factor per element
          in the batch. 0 generates a completely black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Shape:
        - Input: Image/Input to be adjusted in the shape of :math:`(*, N)`.
        - Output: Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustContrast(0.5)(x)
        tensor([[[[0.5000, 0.5000, 0.5000],
                  [0.5000, 0.5000, 0.5000],
                  [0.5000, 0.5000, 0.5000]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.ones(2)
        >>> AdjustContrast(y)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    def __init__(self, contrast_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.contrast_factor: Union[float, Tensor] = contrast_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_contrast(input, self.contrast_factor)

class AdjustHue(Module):
    r"""Adjust hue of an image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.
    The input image is expected to be in the range of [0, 1].

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        hue_factor: How much to shift the hue channel. Should be in [-PI, PI]. PI
          and -PI give complete reversal of hue channel in HSV space in positive and negative
          direction respectively. 0 means no shift. Therefore, both -PI and PI will give an
          image with complementary colors while 0 gives the original image.

    Shape:
        - Input: Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        - Output: Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> AdjustHue(3.141516)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2) * 3.141516
        >>> AdjustHue(y)(x).shape
        torch.Size([2, 3, 3, 3])
    """

    def __init__(self, hue_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.hue_factor: Union[float, Tensor] = hue_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_hue(input, self.hue_factor)

class AdjustSaturation(Module):
    r"""Adjust color saturation of an image.

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        saturation_factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.
        saturation_mode: The mode to adjust saturation.

    Shape:
        - Input: Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        - Output: Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> AdjustSaturation(2.)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2)
        >>> out = AdjustSaturation(y)(x)
        >>> torch.nn.functional.mse_loss(x, out)
        tensor(0.)
    """

    def __init__(self, saturation_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.saturation_factor: Union[float, Tensor] = saturation_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_saturation(input, self.saturation_factor)

class AdjustSaturationWithGraySubtraction(Module):
    r"""Adjust color saturation of an image.

    This implementation aligns PIL. Hence, the output is close to TorchVision.
    The input image is expected to be in the range of [0, 1].

    The input image is expected to be an RGB or gray image in the range of [0, 1].

    Args:
        saturation_factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.
        saturation_mode: The mode to adjust saturation.

    Shape:
        - Input: Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        - Output: Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> AdjustSaturationWithGraySubtraction(2.)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2)
        >>> out = AdjustSaturationWithGraySubtraction(y)(x)
        >>> torch.nn.functional.mse_loss(x, out)
        tensor(0.)
    """

    def __init__(self, saturation_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.saturation_factor: Union[float, Tensor] = saturation_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_saturation_with_gray_subtraction(input, self.saturation_factor)

class AdjustBrightness(Module):
    r"""Adjust Brightness of an image.

    This implementation aligns OpenCV, not PIL. Hence, the output differs from TorchVision.
    The input image is expected to be in the range of [0, 1].

    Args:
        brightness_factor: Brightness adjust factor per element
          in the batch. 0 does not modify the input image while any other number modify the
          brightness.

    Shape:
        - Input: Image/Input to be adjusted in the shape of :math:`(*, N)`.
        - Output: Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustBrightness(1.)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.ones(2)
        >>> AdjustBrightness(y)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    def __init__(self, brightness_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.brightness_factor: Union[float, Tensor] = brightness_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_brightness(input, self.brightness_factor)

class AdjustBrightnessAccumulative(Module):
    r"""Adjust Brightness of an image accumulatively.

    This implementation aligns PIL. Hence, the output is close to TorchVision.
    The input image is expected to be in the range of [0, 1].

    Args:
        brightness_factor: Brightness adjust factor per element
          in the batch. 0 does not modify the input image while any other number modify the
          brightness.

    Shape:
        - Input: Image/Input to be adjusted in the shape of :math:`(*, N)`.
        - Output: Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustBrightnessAccumulative(1.)(x)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.ones(2)
        >>> AdjustBrightnessAccumulative(y)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    def __init__(self, brightness_factor: Union[float, Tensor]) -> None:
        super().__init__()
        self.brightness_factor: Union[float, Tensor] = brightness_factor

    def forward(self, input: Tensor) -> Tensor:
        return adjust_brightness_accumulative(input, self.brightness_factor)

class AdjustSigmoid(Module):
    r"""Adjust the contrast of an image tensor or performs sigmoid correction on the input image tensor.

    The input image is expected to be in the range of [0, 1].

    Reference:
        [1]: Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast Enhancement Functions",
             http://markfairchild.org/PDFs/PAP07.pdf

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        cutoff: The cutoff of sigmoid function.
        gain: The multiplier of sigmoid function.
        inv: If is set to True the function will return the negative sigmoid correction.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> AdjustSigmoid(gain=0)(x)
        tensor([[[[0.5000, 0.5000],
                  [0.5000, 0.5000]]]])
    """

    def __init__(self, cutoff: float = 0.5, gain: float = 10, inv: bool = False) -> None:
        super().__init__()
        self.cutoff: float = cutoff
        self.gain: float = gain
        self.inv: bool = inv

    def forward(self, image: Tensor) -> Tensor:
        return adjust_sigmoid(image, cutoff=self.cutoff, gain=self.gain, inv=self.inv)

