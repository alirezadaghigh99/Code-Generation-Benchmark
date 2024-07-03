class RandomErasing(BaseTransform):
    """RandomErasing operation.

    Random Erasing randomly selects a rectangle region
    in an image and erases its pixels with random values.
    `RandomErasing <https://arxiv.org/abs/1708.04896>`_.

    Required Keys:

    - img
    - gt_bboxes (HorizontalBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (BitmapMasks) (optional)

    Modified Keys:
    - img
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)

    Args:
        n_patches (int or tuple[int, int]): Number of regions to be dropped.
            If it is given as a tuple, number of patches will be randomly
            selected from the closed interval [``n_patches[0]``,
            ``n_patches[1]``].
        ratio (float or tuple[float, float]): The ratio of erased regions.
            It can be ``float`` to use a fixed ratio or ``tuple[float, float]``
            to randomly choose ratio from the interval.
        squared (bool): Whether to erase square region. Defaults to True.
        bbox_erased_thr (float): The threshold for the maximum area proportion
            of the bbox to be erased. When the proportion of the area where the
            bbox is erased is greater than the threshold, the bbox will be
            removed. Defaults to 0.9.
        img_border_value (int or float or tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
    """

    def __init__(
        self,
        n_patches: Union[int, Tuple[int, int]],
        ratio: Union[float, Tuple[float, float]],
        squared: bool = True,
        bbox_erased_thr: float = 0.9,
        img_border_value: Union[int, float, tuple] = 128,
        mask_border_value: int = 0,
        seg_ignore_label: int = 255,
    ) -> None:
        if isinstance(n_patches, tuple):
            assert len(n_patches) == 2 and 0 <= n_patches[0] < n_patches[1]
        else:
            n_patches = (n_patches, n_patches)
        if isinstance(ratio, tuple):
            assert len(ratio) == 2 and 0 <= ratio[0] < ratio[1] <= 1
        else:
            ratio = (ratio, ratio)

        self.n_patches = n_patches
        self.ratio = ratio
        self.squared = squared
        self.bbox_erased_thr = bbox_erased_thr
        self.img_border_value = img_border_value
        self.mask_border_value = mask_border_value
        self.seg_ignore_label = seg_ignore_label

    @cache_randomness
    def _get_patches(self, img_shape: Tuple[int, int]) -> List[list]:
        """Get patches for random erasing."""
        patches = []
        n_patches = np.random.randint(self.n_patches[0], self.n_patches[1] + 1)
        for _ in range(n_patches):
            if self.squared:
                ratio = np.random.random() * (self.ratio[1] -
                                              self.ratio[0]) + self.ratio[0]
                ratio = (ratio, ratio)
            else:
                ratio = (np.random.random() * (self.ratio[1] - self.ratio[0]) +
                         self.ratio[0], np.random.random() *
                         (self.ratio[1] - self.ratio[0]) + self.ratio[0])
            ph, pw = int(img_shape[0] * ratio[0]), int(img_shape[1] * ratio[1])
            px1, py1 = np.random.randint(0,
                                         img_shape[1] - pw), np.random.randint(
                                             0, img_shape[0] - ph)
            px2, py2 = px1 + pw, py1 + ph
            patches.append([px1, py1, px2, py2])
        return np.array(patches)

    def _transform_img(self, results: dict, patches: List[list]) -> None:
        """Random erasing the image."""
        for patch in patches:
            px1, py1, px2, py2 = patch
            results['img'][py1:py2, px1:px2, :] = self.img_border_value

    def _transform_bboxes(self, results: dict, patches: List[list]) -> None:
        """Random erasing the bboxes."""
        bboxes = results['gt_bboxes']
        # TODO: unify the logic by using operators in BaseBoxes.
        assert isinstance(bboxes, HorizontalBoxes)
        bboxes = bboxes.numpy()
        left_top = np.maximum(bboxes[:, None, :2], patches[:, :2])
        right_bottom = np.minimum(bboxes[:, None, 2:], patches[:, 2:])
        wh = np.maximum(right_bottom - left_top, 0)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (
            bboxes[:, 3] - bboxes[:, 1])
        bboxes_erased_ratio = inter_areas.sum(-1) / (bbox_areas + 1e-7)
        valid_inds = bboxes_erased_ratio < self.bbox_erased_thr
        results['gt_bboxes'] = HorizontalBoxes(bboxes[valid_inds])
        results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_inds]
        results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_inds]
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'][valid_inds]

    def _transform_masks(self, results: dict, patches: List[list]) -> None:
        """Random erasing the masks."""
        for patch in patches:
            px1, py1, px2, py2 = patch
            results['gt_masks'].masks[:, py1:py2,
                                      px1:px2] = self.mask_border_value

    def _transform_seg(self, results: dict, patches: List[list]) -> None:
        """Random erasing the segmentation map."""
        for patch in patches:
            px1, py1, px2, py2 = patch
            results['gt_seg_map'][py1:py2, px1:px2] = self.seg_ignore_label

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to erase some regions of image."""
        patches = self._get_patches(results['img_shape'])
        self._transform_img(results, patches)
        if results.get('gt_bboxes', None) is not None:
            self._transform_bboxes(results, patches)
        if results.get('gt_masks', None) is not None:
            self._transform_masks(results, patches)
        if results.get('gt_seg_map', None) is not None:
            self._transform_seg(results, patches)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_patches={self.n_patches}, '
        repr_str += f'ratio={self.ratio}, '
        repr_str += f'squared={self.squared}, '
        repr_str += f'bbox_erased_thr={self.bbox_erased_thr}, '
        repr_str += f'img_border_value={self.img_border_value}, '
        repr_str += f'mask_border_value={self.mask_border_value}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label})'
        return repr_str