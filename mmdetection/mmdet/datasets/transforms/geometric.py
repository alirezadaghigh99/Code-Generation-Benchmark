class Rotate(GeomTransform):
    """Rotate the images, bboxes, masks and segmentation map.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for perform transformation and
            should be in range 0 to 1. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The maximum angle for rotation.
            Defaults to 0.0.
        max_mag (float): The maximum angle for rotation.
            Defaults to 30.0.
        reversal_prob (float): The probability that reverses the rotation
            magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 30.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        assert 0. <= min_mag <= 180., \
            f'min_mag for Rotate should be in range [0,180], got {min_mag}.'
        assert 0. <= max_mag <= 180., \
            f'max_mag for Rotate should be in range [0,180], got {max_mag}.'
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            mask_border_value=mask_border_value,
            seg_ignore_label=seg_ignore_label,
            interpolation=interpolation)

    def _get_homography_matrix(self, results: dict, mag: float) -> np.ndarray:
        """Get the homography matrix for Rotate."""
        img_shape = results['img_shape']
        center = ((img_shape[1] - 1) * 0.5, (img_shape[0] - 1) * 0.5)
        cv2_rotation_matrix = cv2.getRotationMatrix2D(center, -mag, 1.0)
        return np.concatenate(
            [cv2_rotation_matrix,
             np.array([0, 0, 1]).reshape((1, 3))]).astype(np.float32)

    def _transform_img(self, results: dict, mag: float) -> None:
        """Rotate the image."""
        results['img'] = mmcv.imrotate(
            results['img'],
            mag,
            border_value=self.img_border_value,
            interpolation=self.interpolation)

    def _transform_masks(self, results: dict, mag: float) -> None:
        """Rotate the masks."""
        results['gt_masks'] = results['gt_masks'].rotate(
            results['img_shape'],
            mag,
            border_value=self.mask_border_value,
            interpolation=self.interpolation)

    def _transform_seg(self, results: dict, mag: float) -> None:
        """Rotate the segmentation map."""
        results['gt_seg_map'] = mmcv.imrotate(
            results['gt_seg_map'],
            mag,
            border_value=self.seg_ignore_label,
            interpolation='nearest')