class Mosaic(BaseTransform):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 prob: float = 1.0) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'

        log_img_scale(img_scale, skip_square=True, shape_order='wh')
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        if random.uniform(0, 1) > self.prob:
            return results

        assert 'mix_results' in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i,
                                self.img_scale[0] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside(
            [2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape[:2]
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags
        return results

    def _mosaic_combine(
            self, loc: str, center_position_xy: Sequence[float],
            img_shape_wh: Sequence[int]) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str

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

