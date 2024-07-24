class RandAugment(RandomChoice):
    """Rand augmentation.

    This data augmentation is proposed in `RandAugment:
    Practical automated data augmentation with a reduced
    search space <https://arxiv.org/abs/1909.13719>`_.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_bboxes_labels
    - gt_masks
    - gt_ignore_flags
    - gt_seg_map

    Added Keys:

    - homography_matrix

    Args:
        aug_space (List[List[Union[dict, ConfigDict]]]): The augmentation space
            of rand augmentation. Each augmentation transform in ``aug_space``
            is a specific transform, and is composed by several augmentations.
            When RandAugment is called, a random transform in ``aug_space``
            will be selected to augment images. Defaults to aug_space.
        aug_num (int): Number of augmentation to apply equentially.
            Defaults to 2.
        prob (list[float], optional): The probabilities associated with
            each augmentation. The length should be equal to the
            augmentation space and the sum should be 1. If not given,
            a uniform distribution will be assumed. Defaults to None.

    Examples:
        >>> aug_space = [
        >>>     dict(type='Sharpness'),
        >>>     dict(type='ShearX'),
        >>>     dict(type='Color'),
        >>>     ],
        >>> augmentation = RandAugment(aug_space)
        >>> img = np.ones(100, 100, 3)
        >>> gt_bboxes = np.ones(10, 4)
        >>> results = dict(img=img, gt_bboxes=gt_bboxes)
        >>> results = augmentation(results)
    """

    def __init__(self,
                 aug_space: List[Union[dict, ConfigDict]] = RANDAUG_SPACE,
                 aug_num: int = 2,
                 prob: Optional[List[float]] = None) -> None:
        assert isinstance(aug_space, list) and len(aug_space) > 0, \
            'Augmentation space must be a non-empty list.'
        for aug in aug_space:
            assert isinstance(aug, list) and len(aug) == 1, \
                'Each augmentation in aug_space must be a list.'
            for transform in aug:
                assert isinstance(transform, dict) and 'type' in transform, \
                    'Each specific transform must be a dict with key' \
                    ' "type".'
        super().__init__(transforms=aug_space, prob=prob)
        self.aug_space = aug_space
        self.aug_num = aug_num

    @cache_randomness
    def random_pipeline_index(self):
        indices = np.arange(len(self.transforms))
        return np.random.choice(
            indices, self.aug_num, p=self.prob, replace=False)

    def transform(self, results: dict) -> dict:
        """Transform function to use RandAugment.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with RandAugment.
        """
        for idx in self.random_pipeline_index():
            results = self.transforms[idx](results)
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' \
               f'aug_space={self.aug_space}, '\
               f'aug_num={self.aug_num}, ' \
               f'prob={self.prob})'

