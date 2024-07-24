class YOLOv5MixUp(BaseMixImageTransform):
    """MixUp data augmentation for YOLOv5.

    .. code:: text

    The mixup transform steps are as follows:

        1. Another random image is picked by dataset.
        2. Randomly obtain the fusion ratio from the beta distribution,
            then fuse the target
        of the original image and mixup image through this ratio.

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
        alpha (float): parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        beta (float):  parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        pre_transform (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """

    def __init__(self,
                 alpha: float = 32.0,
                 beta: float = 32.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        if use_cached:
            assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
                                           f'but got {max_cached_images}.'
        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)
        self.alpha = alpha
        self.beta = beta

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            int: indexes.
        """
        return random.randint(0, len(dataset))

    def mix_img_transform(self, results: dict) -> dict:
        """YOLOv5 MixUp transform function.

        Args:
            results (dict): Result dict

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']
        ori_img = results['img']
        assert ori_img.shape == retrieve_img.shape

        # Randomly obtain the fusion ratio from the beta distribution,
        # which is around 0.5
        ratio = np.random.beta(self.alpha, self.beta)
        mixup_img = (ori_img * ratio + retrieve_img * (1 - ratio))

        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']

        mixup_gt_bboxes = retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)
        if 'gt_masks' in results:
            assert 'gt_masks' in retrieve_results
            mixup_gt_masks = results['gt_masks'].cat(
                [results['gt_masks'], retrieve_results['gt_masks']])
            results['gt_masks'] = mixup_gt_masks

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags

        return results

class Mosaic9(BaseMixImageTransform):
    """Mosaic9 augmentation.

    Given 9 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                +-------------------------------+------------+
                | pad           |      pad      |            |
                |    +----------+               |            |
                |    |          +---------------+  top_right |
                |    |          |      top      |   image2   |
                |    | top_left |     image1    |            |
                |    |  image8  o--------+------+--------+---+
                |    |          |        |               |   |
                +----+----------+        |     right     |pad|
                |               | center |     image3    |   |
                |     left      | image0 +---------------+---|
                |    image7     |        |               |   |
            +---+-----------+---+--------+               |   |
            |   |  cropped  |            |  bottom_right |pad|
            |   |bottom_left|            |    image4     |   |
            |   |  image6   |   bottom   |               |   |
            +---|-----------+   image5   +---------------+---|
                |    pad    |            |        pad        |
                +-----------+------------+-------------------+

     The mosaic transform steps are as follows:

         1. Get the center image according to the index, and randomly
            sample another 8 images from the custom dataset.
         2. Randomly offset the image after Mosaic

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
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 5 caches for each image suffices for
            randomness. Defaults to 50.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 bbox_clip_border: bool = True,
                 pad_val: Union[float, int] = 114.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 50,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
        if use_cached:
            assert max_cached_images >= 9, 'The length of cache must >= 9, ' \
                                           f'but got {max_cached_images}.'

        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)

        self.img_scale = img_scale
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val

        # intermediate variables
        self._current_img_shape = [0, 0]
        self._center_img_shape = [0, 0]
        self._previous_img_shape = [0, 0]

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> list:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            list: indexes.
        """
        indexes = [random.randint(0, len(dataset)) for _ in range(8)]
        return indexes

    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results

        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []

        img_scale_w, img_scale_h = self.img_scale

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(img_scale_h * 3), int(img_scale_w * 3), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full((int(img_scale_h * 3), int(img_scale_w * 3)),
                                 self.pad_val,
                                 dtype=results['img'].dtype)

        # index = 0 is mean original image
        # len(results['mix_results']) = 8
        loc_strs = ('center', 'top', 'top_right', 'right', 'bottom_right',
                    'bottom', 'bottom_left', 'left', 'top_left')

        results_all = [results, *results['mix_results']]
        for index, results_patch in enumerate(results_all):
            img_i = results_patch['img']
            # keep_ratio resize
            img_i_h, img_i_w = img_i.shape[:2]
            scale_ratio_i = min(img_scale_h / img_i_h, img_scale_w / img_i_w)
            img_i = mmcv.imresize(
                img_i,
                (int(img_i_w * scale_ratio_i), int(img_i_h * scale_ratio_i)))

            paste_coord = self._mosaic_combine(loc_strs[index],
                                               img_i.shape[:2])

            padw, padh = paste_coord[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in paste_coord)
            mosaic_img[y1:y2, x1:x2] = img_i[y1 - padh:, x1 - padw:]

            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)

        # Offset
        offset_x = int(random.uniform(0, img_scale_w))
        offset_y = int(random.uniform(0, img_scale_h))
        mosaic_img = mosaic_img[offset_y:offset_y + 2 * img_scale_h,
                                offset_x:offset_x + 2 * img_scale_w]

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes.translate_([-offset_x, -offset_y])
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * img_scale_h, 2 * img_scale_w])
        else:
            # remove outside bboxes
            inside_inds = mosaic_bboxes.is_inside(
                [2 * img_scale_h, 2 * img_scale_w]).numpy()
            mosaic_bboxes = mosaic_bboxes[inside_inds]
            mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
            mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags
        return results

    def _mosaic_combine(self, loc: str,
                        img_shape_hw: Tuple[int, int]) -> Tuple[int, ...]:
        """Calculate global coordinate of mosaic image.

        Args:
            loc (str): Index for the sub-image.
            img_shape_hw (Sequence[int]): Height and width of sub-image

        Returns:
             paste_coord (tuple): paste corner coordinate in mosaic image.
        """
        assert loc in ('center', 'top', 'top_right', 'right', 'bottom_right',
                       'bottom', 'bottom_left', 'left', 'top_left')

        img_scale_w, img_scale_h = self.img_scale

        self._current_img_shape = img_shape_hw
        current_img_h, current_img_w = self._current_img_shape
        previous_img_h, previous_img_w = self._previous_img_shape
        center_img_h, center_img_w = self._center_img_shape

        if loc == 'center':
            self._center_img_shape = self._current_img_shape
            #  xmin, ymin, xmax, ymax
            paste_coord = img_scale_w, \
                img_scale_h, \
                img_scale_w + current_img_w, \
                img_scale_h + current_img_h
        elif loc == 'top':
            paste_coord = img_scale_w, \
                          img_scale_h - current_img_h, \
                          img_scale_w + current_img_w, \
                          img_scale_h
        elif loc == 'top_right':
            paste_coord = img_scale_w + previous_img_w, \
                          img_scale_h - current_img_h, \
                          img_scale_w + previous_img_w + current_img_w, \
                          img_scale_h
        elif loc == 'right':
            paste_coord = img_scale_w + center_img_w, \
                          img_scale_h, \
                          img_scale_w + center_img_w + current_img_w, \
                          img_scale_h + current_img_h
        elif loc == 'bottom_right':
            paste_coord = img_scale_w + center_img_w, \
                          img_scale_h + previous_img_h, \
                          img_scale_w + center_img_w + current_img_w, \
                          img_scale_h + previous_img_h + current_img_h
        elif loc == 'bottom':
            paste_coord = img_scale_w + center_img_w - current_img_w, \
                          img_scale_h + center_img_h, \
                          img_scale_w + center_img_w, \
                          img_scale_h + center_img_h + current_img_h
        elif loc == 'bottom_left':
            paste_coord = img_scale_w + center_img_w - \
                          previous_img_w - current_img_w, \
                          img_scale_h + center_img_h, \
                          img_scale_w + center_img_w - previous_img_w, \
                          img_scale_h + center_img_h + current_img_h
        elif loc == 'left':
            paste_coord = img_scale_w - current_img_w, \
                          img_scale_h + center_img_h - current_img_h, \
                          img_scale_w, \
                          img_scale_h + center_img_h
        elif loc == 'top_left':
            paste_coord = img_scale_w - current_img_w, \
                          img_scale_h + center_img_h - \
                          previous_img_h - current_img_h, \
                          img_scale_w, \
                          img_scale_h + center_img_h - previous_img_h

        self._previous_img_shape = self._current_img_shape
        #  xmin, ymin, xmax, ymax
        return paste_coord

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str

class YOLOXMixUp(BaseMixImageTransform):
    """MixUp data augmentation for YOLOX.

    .. code:: text

                         mixup transform
                +---------------+--------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                +---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      +-----------------+     |
                |             pad              |
                +------------------------------+

    The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

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
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 ratio_range: Tuple[float, float] = (0.5, 1.5),
                 flip_ratio: float = 0.5,
                 pad_val: float = 114.0,
                 bbox_clip_border: bool = True,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        assert isinstance(img_scale, tuple)
        if use_cached:
            assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
                                           f'but got {max_cached_images}.'
        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch)
        self.img_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            int: indexes.
        """
        return random.randint(0, len(dataset))

    def mix_img_transform(self, results: dict) -> dict:
        """YOLOX MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones((self.img_scale[1], self.img_scale[0], 3),
                              dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.img_scale[::-1], dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.img_scale[1] / retrieve_img.shape[0],
                          self.img_scale[0] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.ones((max(origin_h, target_h), max(
            origin_w, target_w), 3)) * self.pad_val
        padded_img = padded_img.astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes.rescale_([scale_ratio, scale_ratio])
        if self.bbox_clip_border:
            retrieve_gt_bboxes.clip_([origin_h, origin_w])

        if is_filp:
            retrieve_gt_bboxes.flip_([origin_h, origin_w],
                                     direction='horizontal')

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes.translate_([-x_offset, -y_offset])
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes.clip_([target_h, target_w])

        # 8. mix up
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img

        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']

        mixup_gt_bboxes = cp_retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], cp_retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)

        if not self.bbox_clip_border:
            # remove outside bbox
            inside_inds = mixup_gt_bboxes.is_inside([target_h,
                                                     target_w]).numpy()
            mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
            mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]
            mixup_gt_ignore_flags = mixup_gt_ignore_flags[inside_inds]

        if 'gt_keypoints' in results:
            # adjust kps
            retrieve_gt_keypoints = retrieve_results['gt_keypoints']
            retrieve_gt_keypoints.rescale_([scale_ratio, scale_ratio])
            if self.bbox_clip_border:
                retrieve_gt_keypoints.clip_([origin_h, origin_w])

            if is_filp:
                retrieve_gt_keypoints.flip_([origin_h, origin_w],
                                            direction='horizontal')

            # filter
            cp_retrieve_gt_keypoints = retrieve_gt_keypoints.clone()
            cp_retrieve_gt_keypoints.translate_([-x_offset, -y_offset])
            if self.bbox_clip_border:
                cp_retrieve_gt_keypoints.clip_([target_h, target_w])

            # mixup
            mixup_gt_keypoints = cp_retrieve_gt_keypoints.cat(
                (results['gt_keypoints'], cp_retrieve_gt_keypoints), dim=0)
            if not self.bbox_clip_border:
                # remove outside bbox
                mixup_gt_keypoints = mixup_gt_keypoints[inside_inds]
            results['gt_keypoints'] = mixup_gt_keypoints

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_refetch={self.max_refetch}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

