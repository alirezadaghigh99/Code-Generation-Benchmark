class LoadAnnotations(MMCV_LoadAnnotations):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,

                # Used in instance/panoptic segmentation. The segmentation mask
                # of the instance or the information of segments.
                # 1. If list[list[float]], it represents a list of polygons,
                # one for each connected component of the object. Each
                # list[float] is one simple polygon in the format of
                # [x1, y1, ..., xn, yn] (n >= 3). The Xs and Ys are absolute
                # coordinates in unit of pixels.
                # 2. If dict, it represents the per-pixel segmentation mask in
                # COCO's compressed RLE format. The dict should have keys
                # “size” and “counts”.  Can be loaded by pycocotools
                'mask': list[list[float]] or dict,

                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': BaseBoxes(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
        }

    Required Keys:

    - height
    - width
    - instances

      - bbox (optional)
      - bbox_label
      - mask (optional)
      - ignore_flag

    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (bool)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        poly2mask (bool): Whether to convert mask to bitmap. Default: True.
        box_type (str): The box type used to wrap the bboxes. If ``box_type``
            is None, gt_bboxes will keep being np.ndarray. Defaults to 'hbox'.
        reduce_zero_label (bool): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to False.
        ignore_index (int): The label index to be ignored.
            Valid only if reduce_zero_label is true. Defaults is 255.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(
            self,
            with_mask: bool = False,
            poly2mask: bool = True,
            box_type: str = 'hbox',
            # use for semseg
            reduce_zero_label: bool = False,
            ignore_index: int = 255,
            **kwargs) -> None:
        super(LoadAnnotations, self).__init__(**kwargs)
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.box_type = box_type
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _poly2mask(self, mask_ann: Union[list, dict], img_h: int,
                   img_w: int) -> np.ndarray:
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            np.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _process_masks(self, results: dict) -> list:
        """Process gt_masks and filter invalid polygons.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            list: Processed gt_masks.
        """
        gt_masks = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_mask = instance['mask']
            # If the annotation of segmentation mask is invalid,
            # ignore the whole instance.
            if isinstance(gt_mask, list):
                gt_mask = [
                    np.array(polygon) for polygon in gt_mask
                    if len(polygon) % 2 == 0 and len(polygon) >= 6
                ]
                if len(gt_mask) == 0:
                    # ignore this instance and set gt_mask to a fake mask
                    instance['ignore_flag'] = 1
                    gt_mask = [np.zeros(6)]
            elif not self.poly2mask:
                # `PolygonMasks` requires a ploygon of format List[np.array],
                # other formats are invalid.
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and \
                    not (gt_mask.get('counts') is not None and
                         gt_mask.get('size') is not None and
                         isinstance(gt_mask['counts'], (list, str))):
                # if gt_mask is a dict, it should include `counts` and `size`,
                # so that `BitmapMasks` can uncompressed RLE
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            # re-process gt_ignore_flags
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            # fake polygon masks will be ignored in `PackDetInputs`
            gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if results.get('seg_map_path', None) is None:
            return

        img_bytes = get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()

        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = self.ignore_index
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == self.ignore_index -
                            1] = self.ignore_index

        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['ignore_index'] = self.ignore_index

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

class LoadTrackAnnotations(LoadAnnotations):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset. It must load ``instances_ids`` which is only used in the
    tracking tasks. The annotation format is as the following:

    .. code-block:: python
        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],
                # Label of image classification.
                'bbox_label': 1,
                # Used in tracking.
                # Id of instances.
                'instance_id': 100,
                # Used in instance/panoptic segmentation. The segmentation mask
                # of the instance or the information of segments.
                # 1. If list[list[float]], it represents a list of polygons,
                # one for each connected component of the object. Each
                # list[float] is one simple polygon in the format of
                # [x1, y1, ..., xn, yn] (n >= 3). The Xs and Ys are absolute
                # coordinates in unit of pixels.
                # 2. If dict, it represents the per-pixel segmentation mask in
                # COCO's compressed RLE format. The dict should have keys
                # “size” and “counts”.  Can be loaded by pycocotools
                'mask': list[list[float]] or dict,
                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:
    .. code-block:: python
        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': np.ndarray(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
        }

    Required Keys:

    - height (optional)
    - width (optional)
    - instances
      - bbox (optional)
      - bbox_label
      - instance_id (optional)
      - mask (optional)
      - ignore_flag (optional)
    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int32)
    - gt_instances_ids (np.int32)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (np.bool)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        # TODO: use bbox_type
        for instance in results['instances']:
            # The datasets which are only format in evaluation don't have
            # groundtruth boxes.
            if 'bbox' in instance:
                gt_bboxes.append(instance['bbox'])
            if 'ignore_flag' in instance:
                gt_ignore_flags.append(instance['ignore_flag'])

        # TODO: check this case
        if len(gt_bboxes) != len(gt_ignore_flags):
            # There may be no ``gt_ignore_flags`` in some cases, we treat them
            # as all False in order to keep the length of ``gt_bboxes`` and
            # ``gt_ignore_flags`` the same
            gt_ignore_flags = [False] * len(gt_bboxes)

        results['gt_bboxes'] = np.array(
            gt_bboxes, dtype=np.float32).reshape(-1, 4)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_instances_ids(self, results: dict) -> None:
        """Private function to load instances id annotations.

        Args:
            results (dict): Result dict from :obj :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict containing instances id annotations.
        """
        gt_instances_ids = []
        for instance in results['instances']:
            gt_instances_ids.append(instance['instance_id'])
        results['gt_instances_ids'] = np.array(
            gt_instances_ids, dtype=np.int32)

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label, instances id
            and semantic segmentation and keypoints annotations.
        """
        results = super().transform(results)
        self._load_instances_ids(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str

class LoadPanopticAnnotations(LoadAnnotations):
    """Load multiple types of panoptic annotations.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,
                },
                ...
            ]
            'segments_info':
            [
                {
                # id = cls_id + instance_id * INSTANCE_OFFSET
                'id': int,

                # Contiguous category id defined in dataset.
                'category': int

                # Thing flag.
                'is_thing': bool
                },
                ...
            ]

            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': BaseBoxes(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
        }

    Required Keys:

    - height
    - width
    - instances
      - bbox
      - bbox_label
      - ignore_flag
    - segments_info
      - id
      - category
      - is_thing
    - seg_map_path

    Added Keys:

    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (bool)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Defaults to True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        box_type (str): The box mode used to wrap the bboxes.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet >= 3.0.0rc7. Defaults to None.
    """

    def __init__(self,
                 with_bbox: bool = True,
                 with_label: bool = True,
                 with_mask: bool = True,
                 with_seg: bool = True,
                 box_type: str = 'hbox',
                 imdecode_backend: str = 'cv2',
                 backend_args: dict = None) -> None:
        try:
            from panopticapi import utils
        except ImportError:
            raise ImportError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')
        self.rgb2id = utils.rgb2id

        super(LoadPanopticAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            with_keypoints=False,
            box_type=box_type,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)

    def _load_masks_and_semantic_segs(self, results: dict) -> None:
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from ``0`` to
        ``num_things - 1``, the background label is from ``num_things`` to
        ``num_things + num_stuff - 1``, 255 means the ignored label (``VOID``).

        Args:
            results (dict): Result dict from :obj:``mmdet.CustomDataset``.
        """
        # seg_map_path is None, when inference on the dataset without gts.
        if results.get('seg_map_path', None) is None:
            return

        img_bytes = get(
            results['seg_map_path'], backend_args=self.backend_args)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = self.rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for segment_info in results['segments_info']:
            mask = (pan_png == segment_info['id'])
            gt_seg = np.where(mask, segment_info['category'], gt_seg)

            # The legal thing masks
            if segment_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))

        if self.with_mask:
            h, w = results['ori_shape']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks

        if self.with_seg:
            results['gt_seg_map'] = gt_seg

    def transform(self, results: dict) -> dict:
        """Function to load multiple types panoptic annotations.

        Args:
            results (dict): Result dict from :obj:``mmdet.CustomDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask or self.with_seg:
            # The tasks completed by '_load_masks' and '_load_semantic_segs'
            # in LoadAnnotations are merged to one function.
            self._load_masks_and_semantic_segs(results)

        return results

class LoadMultiChannelImageFromFiles(BaseTransform):
    """Load multi-channel images from a list of separate channel files.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'unchanged'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet >= 3.0.0rc7. Defaults to None.
    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = 'unchanged',
        imdecode_backend: str = 'cv2',
        file_client_args: dict = None,
        backend_args: dict = None,
    ) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

    def transform(self, results: dict) -> dict:
        """Transform functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        assert isinstance(results['img_path'], list)
        img = []
        for name in results['img_path']:
            img_bytes = get(name, backend_args=self.backend_args)
            img.append(
                mmcv.imfrombytes(
                    img_bytes,
                    flag=self.color_type,
                    backend=self.imdecode_backend))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'backend_args={self.backend_args})')
        return repr_str

