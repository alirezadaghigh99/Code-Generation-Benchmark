class AppendNormalizedDifferenceIndex(IntensityAugmentationBase2D):
    r"""Append normalized difference index as channel to image tensor.

    Computes the following index:

    .. math::

       \text{NDI} = \frac{A - B}{A + B}

    .. versionadded:: 0.2
    """

    def __init__(self, index_a: int, index_b: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_a: reference band channel index
            index_b: difference band channel index
        """
        super().__init__(p=1)
        self.flags = {'index_a': index_a, 'index_b': index_b}

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        band_a = input[..., flags['index_a'], :, :]
        band_b = input[..., flags['index_b'], :, :]
        ndi = (band_a - band_b) / (band_a + band_b + _EPSILON)
        ndi = torch.unsqueeze(ndi, -3)
        input = torch.cat((input, ndi), dim=-3)
        return input

class AppendTriBandNormalizedDifferenceIndex(IntensityAugmentationBase2D):
    r"""Append normalized difference index involving 3 bands as channel to image tensor.

    Computes the following index:

    .. math::

       \text{TBNDI} = \frac{A - (B + C)}{A + (B + C)}

    .. versionadded:: 0.3
    """

    def __init__(self, index_a: int, index_b: int, index_c: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_a: reference band channel index
            index_b: difference band channel index of component 1
            index_c: difference band channel index of component 2
        """
        super().__init__(p=1)
        self.flags = {'index_a': index_a, 'index_b': index_b, 'index_c': index_c}

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        band_a = input[..., flags['index_a'], :, :]
        band_b = input[..., flags['index_b'], :, :]
        band_c = input[..., flags['index_c'], :, :]
        band_d = band_b + band_c
        tbndi = (band_a - band_d) / (band_a + band_d + _EPSILON)
        tbndi = torch.unsqueeze(tbndi, -3)
        input = torch.cat((input, tbndi), dim=-3)
        return input

