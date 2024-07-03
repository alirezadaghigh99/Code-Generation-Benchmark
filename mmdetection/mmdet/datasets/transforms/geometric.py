class GeomTransform(BaseTransform):
    """Base class for geometric transformations. All geometric transformations
    need to inherit from this base class. ``GeomTransform`` unifies the class
    attributes and class functions of geometric transformations (ShearX,
    ShearY, Rotate, TranslateX, and TranslateY), and records the homography
    matrix.

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
        prob (float): The probability for performing the geometric
            transformation and should be in range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for geometric transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for geometric transformation.
            Defaults to 1.0.
        reversal_prob (float): The probability that reverses the geometric
            transformation magnitude. Should be in range [0,1].
            Defaults to 0.5.
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
                 max_mag: float = 1.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        assert 0 <= prob <= 1.0, f'The probability of the transformation ' \
                                 f'should be in range [0,1], got {prob}.'
        assert level is None or isinstance(level, int), \
            f'The level should be None or type int, got {type(level)}.'
        assert level is None or 0 <= level <= _MAX_LEVEL, \
            f'The level should be in range [0,{_MAX_LEVEL}], got {level}.'
        assert isinstance(min_mag, float), \
            f'min_mag should be type float, got {type(min_mag)}.'
        assert isinstance(max_mag, float), \
            f'max_mag should be type float, got {type(max_mag)}.'
        assert min_mag <= max_mag, \
            f'min_mag should smaller than max_mag, ' \
            f'got min_mag={min_mag} and max_mag={max_mag}'
        assert isinstance(reversal_prob, float), \
            f'reversal_prob should be type float, got {type(max_mag)}.'
        assert 0 <= reversal_prob <= 1.0, \
            f'The reversal probability of the transformation magnitude ' \
            f'should be type float, got {type(reversal_prob)}.'
        if isinstance(img_border_value, (float, int)):
            img_border_value = tuple([float(img_border_value)] * 3)
        elif isinstance(img_border_value, tuple):
            assert len(img_border_value) == 3, \
                f'img_border_value as tuple must have 3 elements, ' \
                f'got {len(img_border_value)}.'
            img_border_value = tuple([float(val) for val in img_border_value])
        else:
            raise ValueError(
                'img_border_value must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_border_value]), 'all ' \
            'elements of img_border_value should between range [0,255].' \
            f'got {img_border_value}.'
        self.prob = prob
        self.level = level
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.reversal_prob = reversal_prob
        self.img_border_value = img_border_value
        self.mask_border_value = mask_border_value
        self.seg_ignore_label = seg_ignore_label
        self.interpolation = interpolation

    def _transform_img(self, results: dict, mag: float) -> None:
        """Transform the image."""
        pass

    def _transform_masks(self, results: dict, mag: float) -> None:
        """Transform the masks."""
        pass

    def _transform_seg(self, results: dict, mag: float) -> None:
        """Transform the segmentation map."""
        pass

    def _get_homography_matrix(self, results: dict, mag: float) -> np.ndarray:
        """Get the homography matrix for the geometric transformation."""
        return np.eye(3, dtype=np.float32)

    def _transform_bboxes(self, results: dict, mag: float) -> None:
        """Transform the bboxes."""
        results['gt_bboxes'].project_(self.homography_matrix)
        results['gt_bboxes'].clip_(results['img_shape'])

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the geometric transformation."""
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = self.homography_matrix
        else:
            results['homography_matrix'] = self.homography_matrix @ results[
                'homography_matrix']

    @cache_randomness
    def _random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.prob

    @cache_randomness
    def _get_mag(self):
        """Get the magnitude of the transform."""
        mag = level_to_mag(self.level, self.min_mag, self.max_mag)
        return -mag if np.random.rand() > self.reversal_prob else mag

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function for images, bounding boxes, masks and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """

        if self._random_disable():
            return results
        mag = self._get_mag()
        self.homography_matrix = self._get_homography_matrix(results, mag)
        self._record_homography_matrix(results)
        self._transform_img(results, mag)
        if results.get('gt_bboxes', None) is not None:
            self._transform_bboxes(results, mag)
        if results.get('gt_masks', None) is not None:
            self._transform_masks(results, mag)
        if results.get('gt_seg_map', None) is not None:
            self._transform_seg(results, mag)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'level={self.level}, '
        repr_str += f'min_mag={self.min_mag}, '
        repr_str += f'max_mag={self.max_mag}, '
        repr_str += f'reversal_prob={self.reversal_prob}, '
        repr_str += f'img_border_value={self.img_border_value}, '
        repr_str += f'mask_border_value={self.mask_border_value}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str