class MinResize(object):
    """Transform that resizes the PIL image or torch Tensor, if necessary, so
    that its minimum dimensions are at least the specified size.

    Args:
        min_output_size: desired minimum output dimensions. Can either be a
            ``(min_height, min_width)`` tuple or a single ``min_dim``
        interpolation (None): optional interpolation mode. Passed directly to
            :func:`torchvision:torchvision.transforms.functional.resize`
    """

    def __init__(self, min_output_size, interpolation=None):
        if isinstance(min_output_size, int):
            min_output_size = (min_output_size, min_output_size)

        self.min_output_size = min_output_size
        self.interpolation = interpolation

        self._kwargs = {}
        if interpolation is not None:
            self._kwargs["interpolation"] = interpolation

    def __call__(self, pil_image_or_tensor):
        if isinstance(pil_image_or_tensor, torch.Tensor):
            h, w = list(pil_image_or_tensor.size())[-2:]
        else:
            w, h = pil_image_or_tensor.size

        minh, minw = self.min_output_size

        if h >= minh and w >= minw:
            return pil_image_or_tensor

        alpha = max(minh / h, minw / w)
        size = (int(round(alpha * h)), int(round(alpha * w)))
        return F.resize(pil_image_or_tensor, size, **self._kwargs)

class MaxResize(object):
    """Transform that resizes the PIL image or torch Tensor, if necessary, so
    that its maximum dimensions are at most the specified size.

    Args:
        max_output_size: desired maximum output dimensions. Can either be a
            ``(max_height, max_width)`` tuple or a single ``max_dim``
        interpolation (None): optional interpolation mode. Passed directly to
            :func:`torchvision:torchvision.transforms.functional.resize`
    """

    def __init__(self, max_output_size, interpolation=None):
        if isinstance(max_output_size, int):
            max_output_size = (max_output_size, max_output_size)

        self.max_output_size = max_output_size
        self.interpolation = interpolation

        self._kwargs = {}
        if interpolation is not None:
            self._kwargs["interpolation"] = interpolation

    def __call__(self, pil_image_or_tensor):
        if isinstance(pil_image_or_tensor, torch.Tensor):
            h, w = list(pil_image_or_tensor.size())[-2:]
        else:
            w, h = pil_image_or_tensor.size

        maxh, maxw = self.max_output_size

        if h <= maxh and w <= maxw:
            return pil_image_or_tensor

        alpha = min(maxh / h, maxw / w)
        size = (int(round(alpha * h)), int(round(alpha * w)))
        return F.resize(pil_image_or_tensor, size, **self._kwargs)

