class AutoContrast(ColorTransform):
    """Auto adjust image contrast.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing AutoContrast should
             be in range [0, 1]. Defaults to 1.0.
        level (int, optional): No use for AutoContrast transformation.
            Defaults to None.
        min_mag (float): No use for AutoContrast transformation.
            Defaults to 0.1.
        max_mag (float): No use for AutoContrast transformation.
            Defaults to 1.9.
    """

    def _transform_img(self, results: dict, mag: float) -> None:
        """Auto adjust image contrast."""
        img = results['img']
        results['img'] = mmcv.auto_contrast(img).astype(img.dtype)