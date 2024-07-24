class LoadAnnotations(MMDET_LoadAnnotations):
    """Because the yolo series does not need to consider ignore bboxes for the
    time being, in order to speed up the pipeline, it can be excluded in
    advance.

    Args:
        mask2bbox (bool): Whether to use mask annotation to get bbox.
            Defaults to False.
        poly2mask (bool): Whether to transform the polygons to bitmaps.
            Defaults to False.
        merge_polygons (bool): Whether to merge polygons into one polygon.
            If merged, the storage structure is simpler and training is more
            effcient, especially if the mask inside a bbox is divided into
            multiple polygons. Defaults to True.
    """

    def __init__(self,
                 mask2bbox: bool = False,
                 poly2mask: bool = False,
                 merge_polygons: bool = True,
                 **kwargs):
        self.mask2bbox = mask2bbox
        self.merge_polygons = merge_polygons
        assert not poly2mask, 'Does not support BitmapMasks considering ' \
                              'that bitmap consumes more memory.'
        super().__init__(poly2mask=poly2mask, **kwargs)
        if self.mask2bbox:
            assert self.with_mask, 'Using mask2bbox requires ' \
                                   'with_mask is True.'
        self._mask_ignore_flag = None

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        if self.mask2bbox:
            self._load_masks(results)
            if self.with_label:
                self._load_labels(results)
                self._update_mask_ignore_data(results)
            gt_bboxes = results['gt_masks'].get_bboxes(dst_type='hbox')
            results['gt_bboxes'] = gt_bboxes
        elif self.with_keypoints:
            self._load_kps(results)
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(
                results.get('bbox', []), dtype=torch.float32)
        else:
            results = super().transform(results)
            self._update_mask_ignore_data(results)
        return results

    def _update_mask_ignore_data(self, results: dict) -> None:
        if 'gt_masks' not in results:
            return

        if 'gt_bboxes_labels' in results and len(
                results['gt_bboxes_labels']) != len(results['gt_masks']):
            assert len(results['gt_bboxes_labels']) == len(
                self._mask_ignore_flag)
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                self._mask_ignore_flag]

        if 'gt_bboxes' in results and len(results['gt_bboxes']) != len(
                results['gt_masks']):
            assert len(results['gt_bboxes']) == len(self._mask_ignore_flag)
            results['gt_bboxes'] = results['gt_bboxes'][self._mask_ignore_flag]

    def _load_bboxes(self, results: dict):
        """Private function to load bounding box annotations.
        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes.append(instance['bbox'])
                gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)

    def _load_labels(self, results: dict):
        """Private function to load label annotations.

        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        gt_masks = []
        gt_ignore_flags = []
        self._mask_ignore_flag = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                if 'mask' in instance:
                    gt_mask = instance['mask']
                    if isinstance(gt_mask, list):
                        gt_mask = [
                            np.array(polygon) for polygon in gt_mask
                            if len(polygon) % 2 == 0 and len(polygon) >= 6
                        ]
                        if len(gt_mask) == 0:
                            # ignore
                            self._mask_ignore_flag.append(0)
                        else:
                            if len(gt_mask) > 1 and self.merge_polygons:
                                gt_mask = self.merge_multi_segment(gt_mask)
                            gt_masks.append(gt_mask)
                            gt_ignore_flags.append(instance['ignore_flag'])
                            self._mask_ignore_flag.append(1)
                    else:
                        raise NotImplementedError(
                            'Only supports mask annotations in polygon '
                            'format currently')
                else:
                    # TODO: Actually, gt with bbox and without mask needs
                    #  to be retained
                    self._mask_ignore_flag.append(0)
        self._mask_ignore_flag = np.array(self._mask_ignore_flag, dtype=bool)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        h, w = results['ori_shape']
        gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def merge_multi_segment(self,
                            gt_masks: List[np.ndarray]) -> List[np.ndarray]:
        """Merge multi segments to one list.

        Find the coordinates with min distance between each segment,
        then connect these coordinates with one thin line to merge all
        segments into one.
        Args:
            gt_masks(List(np.array)):
                original segmentations in coco's json file.
                like [segmentation1, segmentation2,...],
                each segmentation is a list of coordinates.
        Return:
            gt_masks(List(np.array)): merged gt_masks
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in gt_masks]
        idx_list = [[] for _ in range(len(gt_masks))]

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = self.min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        # first round: first to end, i.e. A->B(partial)->C
        # second round: end to first, i.e. C->B(remaining)-A
        for k in range(2):
            # forward first round
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]
                    # add the idx[0] point for connect next segment
                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate(
                        [segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    # deal with the middle segment
                    # Note that in the first round, only partial segment
                    # are appended.
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])
            # forward second round
            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    # deal with the middle segment
                    # append the remaining points
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return [np.concatenate(s).reshape(-1, )]

    def min_index(self, arr1: np.ndarray, arr2: np.ndarray) -> Tuple[int, int]:
        """Find a pair of indexes with the shortest distance.

        Args:
            arr1: (N, 2).
            arr2: (M, 2).
        Return:
            tuple: a pair of indexes.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :])**2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        results['height'] = results['img_shape'][0]
        results['width'] = results['img_shape'][1]
        num_instances = len(results.get('bbox', []))

        if num_instances == 0:
            results['keypoints'] = np.empty(
                (0, len(results['flip_indices']), 2), dtype=np.float32)
            results['keypoints_visible'] = np.empty(
                (0, len(results['flip_indices'])), dtype=np.int32)
            results['category_id'] = []

        results['gt_keypoints'] = Keypoints(
            keypoints=results['keypoints'],
            keypoints_visible=results['keypoints_visible'],
            flip_indices=results['flip_indices'],
        )

        results['gt_ignore_flags'] = np.array([False] * num_instances)
        results['gt_bboxes_labels'] = np.array(results['category_id']) - 1

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'mask2bbox={self.mask2bbox}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

class YOLOv5KeepRatioResize(MMDET_Resize):
    """Resize images & bbox(if existed).

    This transform resizes the input image according to ``scale``.
    Bboxes (if existed) are then resized with the same scale factor.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)
    - scale (float)

    Added Keys:

    - scale_factor (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 keep_ratio: bool = True,
                 **kwargs):
        assert keep_ratio is True
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

    @staticmethod
    def _get_rescale_ratio(old_size: Tuple[int, int],
                           scale: Union[float, Tuple[int]]) -> float:
        """Calculate the ratio for rescaling.

        Args:
            old_size (tuple[int]): The old size (w, h) of image.
            scale (float | tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by
                this factor, else if it is a tuple of 2 integers, then
                the image will be rescaled as large as possible within
                the scale.

        Returns:
            float: The resize ratio.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))
        else:
            raise TypeError('Scale must be a number or tuple of int, '
                            f'but got {type(scale)}')

        return scale_factor

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        assert self.keep_ratio is True

        if results.get('img', None) is not None:
            image = results['img']
            original_h, original_w = image.shape[:2]
            ratio = self._get_rescale_ratio((original_h, original_w),
                                            self.scale)

            if ratio != 1:
                # resize image according to the shape
                # NOTE: We are currently testing on COCO that modifying
                # this code will not affect the results.
                # If you find that it has an effect on your results,
                # please feel free to contact us.
                image = mmcv.imresize(
                    img=image,
                    size=(int(original_w * ratio), int(original_h * ratio)),
                    interpolation='area' if ratio < 1 else 'bilinear',
                    backend=self.backend)

            resized_h, resized_w = image.shape[:2]
            scale_ratio_h = resized_h / original_h
            scale_ratio_w = resized_w / original_w
            scale_factor = (scale_ratio_w, scale_ratio_h)

            results['img'] = image
            results['img_shape'] = image.shape[:2]
            results['scale_factor'] = scale_factor

class YOLOv5HSVRandomAug(BaseTransform):
    """Apply HSV augmentation to image sequentially.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta ([int, float]): delta of hue. Defaults to 0.015.
        saturation_delta ([int, float]): delta of saturation. Defaults to 0.7.
        value_delta ([int, float]): delta of value. Defaults to 0.4.
    """

    def __init__(self,
                 hue_delta: Union[int, float] = 0.015,
                 saturation_delta: Union[int, float] = 0.7,
                 value_delta: Union[int, float] = 0.4):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def transform(self, results: dict) -> dict:
        """The HSV augmentation transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        hsv_gains = \
            random.uniform(-1, 1, 3) * \
            [self.hue_delta, self.saturation_delta, self.value_delta] + 1
        hue, sat, val = cv2.split(
            cv2.cvtColor(results['img'], cv2.COLOR_BGR2HSV))

        table_list = np.arange(0, 256, dtype=hsv_gains.dtype)
        lut_hue = ((table_list * hsv_gains[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(table_list * hsv_gains[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(table_list * hsv_gains[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat,
                                            lut_sat), cv2.LUT(val, lut_val)))
        results['img'] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str

class PPYOLOERandomDistort(BaseTransform):
    """Random hue, saturation, contrast and brightness distortion.

    Required Keys:

    - img

    Modified Keys:

    - img (np.float32)

    Args:
        hue_cfg (dict): Hue settings. Defaults to dict(min=-18,
            max=18, prob=0.5).
        saturation_cfg (dict): Saturation settings. Defaults to dict(
            min=0.5, max=1.5, prob=0.5).
        contrast_cfg (dict): Contrast settings. Defaults to dict(
            min=0.5, max=1.5, prob=0.5).
        brightness_cfg (dict): Brightness settings. Defaults to dict(
            min=0.5, max=1.5, prob=0.5).
        num_distort_func (int): The number of distort function. Defaults
            to 4.
    """

    def __init__(self,
                 hue_cfg: dict = dict(min=-18, max=18, prob=0.5),
                 saturation_cfg: dict = dict(min=0.5, max=1.5, prob=0.5),
                 contrast_cfg: dict = dict(min=0.5, max=1.5, prob=0.5),
                 brightness_cfg: dict = dict(min=0.5, max=1.5, prob=0.5),
                 num_distort_func: int = 4):
        self.hue_cfg = hue_cfg
        self.saturation_cfg = saturation_cfg
        self.contrast_cfg = contrast_cfg
        self.brightness_cfg = brightness_cfg
        self.num_distort_func = num_distort_func
        assert 0 < self.num_distort_func <= 4, \
            'num_distort_func must > 0 and <= 4'
        for cfg in [
                self.hue_cfg, self.saturation_cfg, self.contrast_cfg,
                self.brightness_cfg
        ]:
            assert 0. <= cfg['prob'] <= 1., 'prob must >=0 and <=1'

    def transform_hue(self, results):
        """Transform hue randomly."""
        if random.uniform(0., 1.) >= self.hue_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.hue_cfg['min'], self.hue_cfg['max'])
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        delta_iq = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        rgb2yiq_matrix = np.array([[0.114, 0.587, 0.299],
                                   [-0.321, -0.274, 0.596],
                                   [0.311, -0.523, 0.211]])
        yiq2rgb_matric = np.array([[1.0, -1.107, 1.705], [1.0, -0.272, -0.647],
                                   [1.0, 0.956, 0.621]])
        t = np.dot(np.dot(yiq2rgb_matric, delta_iq), rgb2yiq_matrix).T
        img = np.dot(img, t)
        results['img'] = img
        return results

    def transform_saturation(self, results):
        """Transform saturation randomly."""
        if random.uniform(0., 1.) >= self.saturation_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.saturation_cfg['min'],
                               self.saturation_cfg['max'])

        # convert bgr img to gray img
        gray = img * np.array([[[0.114, 0.587, 0.299]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        results['img'] = img
        return results

    def transform_contrast(self, results):
        """Transform contrast randomly."""
        if random.uniform(0., 1.) >= self.contrast_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.contrast_cfg['min'],
                               self.contrast_cfg['max'])
        img *= delta
        results['img'] = img
        return results

    def transform_brightness(self, results):
        """Transform brightness randomly."""
        if random.uniform(0., 1.) >= self.brightness_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.brightness_cfg['min'],
                               self.brightness_cfg['max'])
        img += delta
        results['img'] = img
        return results

    def transform(self, results: dict) -> dict:
        """The hue, saturation, contrast and brightness distortion function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        results['img'] = results['img'].astype(np.float32)

        functions = [
            self.transform_brightness, self.transform_contrast,
            self.transform_saturation, self.transform_hue
        ]
        distortions = random.permutation(functions)[:self.num_distort_func]
        for func in distortions:
            results = func(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(hue_cfg={self.hue_cfg}, '
        repr_str += f'saturation_cfg={self.saturation_cfg}, '
        repr_str += f'contrast_cfg={self.contrast_cfg}, '
        repr_str += f'brightness_cfg={self.brightness_cfg}, '
        repr_str += f'num_distort_func={self.num_distort_func})'
        return repr_str

class LetterResize(MMDET_Resize):
    """Resize and pad image while meeting stride-multiple constraints.

    Required Keys:

    - img (np.uint8)
    - batch_shape (np.int64) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)

    Added Keys:
    - pad_param (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
        pad_val (dict): Padding value. Defaults to dict(img=0, seg=255).
        use_mini_pad (bool): Whether using minimum rectangle padding.
            Defaults to True
        stretch_only (bool): Whether stretch to the specified size directly.
            Defaults to False
        allow_scale_up (bool): Allow scale up when ratio > 1. Defaults to True
        half_pad_param (bool): If set to True, left and right pad_param will
            be given by dividing padding_h by 2. If set to False, pad_param is
            in int format. We recommend setting this to False for object
            detection tasks, and True for instance segmentation tasks.
            Default to False.
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 pad_val: dict = dict(img=0, mask=0, seg=255),
                 use_mini_pad: bool = False,
                 stretch_only: bool = False,
                 allow_scale_up: bool = True,
                 half_pad_param: bool = False,
                 **kwargs):
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

        self.pad_val = pad_val
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(
            pad_val, dict), f'pad_val must be dict, but got {type(pad_val)}'

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up
        self.half_pad_param = half_pad_param

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        image = results.get('img', None)
        if image is None:
            return

        # Use batch_shape if a batch_shape policy is configured
        if 'batch_shape' in results:
            scale = tuple(results['batch_shape'])  # hw
        else:
            scale = self.scale[::-1]  # wh -> hw

        image_shape = image.shape[:2]  # height, width

        # Scale ratio (new / old)
        ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

        # only scale down, do not scale up (for better test mAP)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)

        ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

        # compute the best size of the image
        no_pad_shape = (int(round(image_shape[0] * ratio[0])),
                        int(round(image_shape[1] * ratio[1])))

        # padding height & width
        padding_h, padding_w = [
            scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]
        ]
        if self.use_mini_pad:
            # minimum rectangle padding
            padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)

        elif self.stretch_only:
            # stretch to the specified size directly
            padding_h, padding_w = 0.0, 0.0
            no_pad_shape = (scale[0], scale[1])
            ratio = [scale[0] / image_shape[0],
                     scale[1] / image_shape[1]]  # height, width ratios

        if image_shape != no_pad_shape:
            # compare with no resize and padding size
            image = mmcv.imresize(
                image, (no_pad_shape[1], no_pad_shape[0]),
                interpolation=self.interpolation,
                backend=self.backend)

        scale_factor = (no_pad_shape[1] / image_shape[1],
                        no_pad_shape[0] / image_shape[0])

        if 'scale_factor' in results:
            results['scale_factor_origin'] = results['scale_factor']
        results['scale_factor'] = scale_factor

        # padding
        top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
            round(padding_w // 2 - 0.1))
        bottom_padding = padding_h - top_padding
        right_padding = padding_w - left_padding

        padding_list = [
            top_padding, bottom_padding, left_padding, right_padding
        ]
        if top_padding != 0 or bottom_padding != 0 or \
                left_padding != 0 or right_padding != 0:

            pad_val = self.pad_val.get('img', 0)
            if isinstance(pad_val, int) and image.ndim == 3:
                pad_val = tuple(pad_val for _ in range(image.shape[2]))

            image = mmcv.impad(
                img=image,
                padding=(padding_list[2], padding_list[0], padding_list[3],
                         padding_list[1]),
                pad_val=pad_val,
                padding_mode='constant')

        results['img'] = image
        results['img_shape'] = image.shape
        if 'pad_param' in results:
            results['pad_param_origin'] = results['pad_param'] * \
                                          np.repeat(ratio, 2)

        if self.half_pad_param:
            results['pad_param'] = np.array(
                [padding_h / 2, padding_h / 2, padding_w / 2, padding_w / 2],
                dtype=np.float32)
        else:
            # We found in object detection, using padding list with
            # int type can get higher mAP.
            results['pad_param'] = np.array(padding_list, dtype=np.float32)

    def _resize_masks(self, results: dict):
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is None:
            return

        gt_masks = results['gt_masks']
        assert isinstance(
            gt_masks, PolygonMasks
        ), f'Only supports PolygonMasks, but got {type(gt_masks)}'

        # resize the gt_masks
        gt_mask_h = results['gt_masks'].height * results['scale_factor'][1]
        gt_mask_w = results['gt_masks'].width * results['scale_factor'][0]
        gt_masks = results['gt_masks'].resize(
            (int(round(gt_mask_h)), int(round(gt_mask_w))))

        top_padding, _, left_padding, _ = results['pad_param']
        if int(left_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=results['img_shape'][:2],
                offset=int(left_padding),
                direction='horizontal')
        if int(top_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=results['img_shape'][:2],
                offset=int(top_padding),
                direction='vertical')
        results['gt_masks'] = gt_masks

    def _resize_bboxes(self, results: dict):
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is None:
            return
        results['gt_bboxes'].rescale_(results['scale_factor'])

        if len(results['pad_param']) != 4:
            return
        results['gt_bboxes'].translate_(
            (results['pad_param'][2], results['pad_param'][0]))

        if self.clip_object_border:
            results['gt_bboxes'].clip_(results['img_shape'])

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if 'scale_factor_origin' in results:
            scale_factor_origin = results.pop('scale_factor_origin')
            results['scale_factor'] = (results['scale_factor'][0] *
                                       scale_factor_origin[0],
                                       results['scale_factor'][1] *
                                       scale_factor_origin[1])
        if 'pad_param_origin' in results:
            pad_param_origin = results.pop('pad_param_origin')
            results['pad_param'] += pad_param_origin
        return results

