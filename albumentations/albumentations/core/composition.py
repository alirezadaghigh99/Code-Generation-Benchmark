class Compose(BaseCompose, HubMixin):
    """Compose transforms and handle all transformations regarding bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
        is_check_shapes (bool): If True shapes consistency of images/mask/masks would be checked on each call. If you
            would like to disable this check - pass False (do it only if you are sure in your data consistency).
        strict (bool): If True, unknown keys will raise an error. If False, unknown keys will be ignored. Default: True.
        return_params (bool): if True returns params of each applied transform
        save_key (str): key to save applied params, default is 'applied_params'

    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: dict[str, Any] | BboxParams | None = None,
        keypoint_params: dict[str, Any] | KeypointParams | None = None,
        additional_targets: dict[str, str] | None = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        strict: bool = True,
        return_params: bool = False,
        save_key: str = "applied_params",
    ):
        super().__init__(transforms, p)

        if bbox_params:
            if isinstance(bbox_params, dict):
                b_params = BboxParams(**bbox_params)
            elif isinstance(bbox_params, BboxParams):
                b_params = bbox_params
            else:
                msg = "unknown format of bbox_params, please use `dict` or `BboxParams`"
                raise ValueError(msg)
            self.processors["bboxes"] = BboxProcessor(b_params)

        if keypoint_params:
            if isinstance(keypoint_params, dict):
                k_params = KeypointParams(**keypoint_params)
            elif isinstance(keypoint_params, KeypointParams):
                k_params = keypoint_params
            else:
                msg = "unknown format of keypoint_params, please use `dict` or `KeypointParams`"
                raise ValueError(msg)
            self.processors["keypoints"] = KeypointsProcessor(k_params)

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)
        if not self.transforms:  # if no transforms -> do nothing, all keys will be available
            self._available_keys.update(AVAILABLE_KEYS)

        self.is_check_args = True
        self.strict = strict

        self.is_check_shapes = is_check_shapes
        self.check_each_transform = tuple(  # processors that checks after each transform
            proc for proc in self.processors.values() if getattr(proc.params, "check_each_transform", False)
        )
        self._set_check_args_for_transforms(self.transforms)

        self.return_params = return_params
        if return_params:
            self.save_key = save_key
            self._available_keys.add(save_key)
            self._transforms_dict = get_transforms_dict(self.transforms)
            self.set_deterministic(True, save_key=save_key)

    def _set_check_args_for_transforms(self, transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                self._set_check_args_for_transforms(transform.transforms)
                transform.check_each_transform = self.check_each_transform
                transform.processors = self.processors
            if isinstance(transform, Compose):
                transform.disable_check_args_private()

    def disable_check_args_private(self) -> None:
        self.is_check_args = False
        self.strict = False
        self.main_compose = False

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)

        if not isinstance(force_apply, (bool, int)):
            msg = "force_apply must have bool or int type"
            raise TypeError(msg)

        if self.return_params and self.main_compose:
            data[self.save_key] = OrderedDict()

        need_to_run = force_apply or random.random() < self.p
        if not need_to_run:
            return data

        self.preprocess(data)

        for t in self.transforms:
            data = t(**data)
            data = self.check_data_post_transform(data)

        return self.postprocess(data)

    def run_with_params(self, *, params: dict[int, dict[str, Any]], **data: Any) -> dict[str, Any]:
        """Run transforms with given parameters. Available only for Compose with `return_params=True`."""
        if self._transforms_dict is None:
            raise RuntimeError("`run_with_params` is not available for Compose with `return_params=False`.")

        self.preprocess(data)

        for tr_id, param in params.items():
            tr = self._transforms_dict[tr_id]
            data = tr.apply_with_params(param, **data)
            data = self.check_data_post_transform(data)

        return self.postprocess(data)

    def preprocess(self, data: Any) -> None:
        if self.strict:
            for data_name in data:
                if data_name not in self._available_keys and data_name not in MASK_KEYS:
                    msg = f"Key {data_name} is not in available keys."
                    raise ValueError(msg)
        if self.is_check_args:
            self._check_args(**data)
        if self.main_compose:
            for p in self.processors.values():
                p.ensure_data_valid(data)
            for p in self.processors.values():
                p.preprocess(data)

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.main_compose:
            data = Compose._make_targets_contiguous(data)  # ensure output targets are contiguous
            for p in self.processors.values():
                p.postprocess(data)
        return data

    def to_dict_private(self) -> dict[str, Any]:
        dictionary = super().to_dict_private()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params.to_dict_private() if bbox_processor else None,
                "keypoint_params": (keypoints_processor.params.to_dict_private() if keypoints_processor else None),
                "additional_targets": self.additional_targets,
                "is_check_shapes": self.is_check_shapes,
            },
        )
        return dictionary

    def get_dict_with_id(self) -> dict[str, Any]:
        dictionary = super().get_dict_with_id()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params.to_dict_private() if bbox_processor else None,
                "keypoint_params": (keypoints_processor.params.to_dict_private() if keypoints_processor else None),
                "additional_targets": self.additional_targets,
                "params": None,
                "is_check_shapes": self.is_check_shapes,
            },
        )
        return dictionary

    def _check_args(self, **kwargs: Any) -> None:
        shapes = []

        for data_name, data in kwargs.items():
            internal_data_name = self._additional_targets.get(data_name, data_name)
            if internal_data_name in CHECKED_SINGLE:
                if not isinstance(data, np.ndarray):
                    raise TypeError(f"{data_name} must be numpy array type")
                shapes.append(data.shape[:2])
            if internal_data_name in CHECKED_MULTI and data is not None and len(data):
                if not isinstance(data[0], np.ndarray):
                    raise TypeError(f"{data_name} must be list of numpy arrays")
                shapes.append(data[0].shape[:2])
            if internal_data_name in CHECK_BBOX_PARAM and self.processors.get("bboxes") is None:
                msg = "bbox_params must be specified for bbox transformations"
                raise ValueError(msg)

            if internal_data_name in CHECK_KEYPOINTS_PARAM and self.processors.get("keypoints") is None:
                msg = "keypoints_params must be specified for keypoint transformations"
                raise ValueError(msg)

        if self.is_check_shapes and shapes and shapes.count(shapes[0]) != len(shapes):
            msg = (
                "Height and Width of image, mask or masks should be equal. You can disable shapes check "
                "by setting a parameter is_check_shapes=False of Compose class (do it only if you are sure "
                "about your data consistency)."
            )
            raise ValueError(msg)

    @staticmethod
    def _make_targets_contiguous(data: Any) -> dict[str, Any]:
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                result[key] = np.ascontiguousarray(value)
            else:
                result[key] = value

        return resultclass Compose(BaseCompose, HubMixin):
    """Compose transforms and handle all transformations regarding bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
        is_check_shapes (bool): If True shapes consistency of images/mask/masks would be checked on each call. If you
            would like to disable this check - pass False (do it only if you are sure in your data consistency).
        strict (bool): If True, unknown keys will raise an error. If False, unknown keys will be ignored. Default: True.
        return_params (bool): if True returns params of each applied transform
        save_key (str): key to save applied params, default is 'applied_params'

    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: dict[str, Any] | BboxParams | None = None,
        keypoint_params: dict[str, Any] | KeypointParams | None = None,
        additional_targets: dict[str, str] | None = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        strict: bool = True,
        return_params: bool = False,
        save_key: str = "applied_params",
    ):
        super().__init__(transforms, p)

        if bbox_params:
            if isinstance(bbox_params, dict):
                b_params = BboxParams(**bbox_params)
            elif isinstance(bbox_params, BboxParams):
                b_params = bbox_params
            else:
                msg = "unknown format of bbox_params, please use `dict` or `BboxParams`"
                raise ValueError(msg)
            self.processors["bboxes"] = BboxProcessor(b_params)

        if keypoint_params:
            if isinstance(keypoint_params, dict):
                k_params = KeypointParams(**keypoint_params)
            elif isinstance(keypoint_params, KeypointParams):
                k_params = keypoint_params
            else:
                msg = "unknown format of keypoint_params, please use `dict` or `KeypointParams`"
                raise ValueError(msg)
            self.processors["keypoints"] = KeypointsProcessor(k_params)

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)
        if not self.transforms:  # if no transforms -> do nothing, all keys will be available
            self._available_keys.update(AVAILABLE_KEYS)

        self.is_check_args = True
        self.strict = strict

        self.is_check_shapes = is_check_shapes
        self.check_each_transform = tuple(  # processors that checks after each transform
            proc for proc in self.processors.values() if getattr(proc.params, "check_each_transform", False)
        )
        self._set_check_args_for_transforms(self.transforms)

        self.return_params = return_params
        if return_params:
            self.save_key = save_key
            self._available_keys.add(save_key)
            self._transforms_dict = get_transforms_dict(self.transforms)
            self.set_deterministic(True, save_key=save_key)

    def _set_check_args_for_transforms(self, transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                self._set_check_args_for_transforms(transform.transforms)
                transform.check_each_transform = self.check_each_transform
                transform.processors = self.processors
            if isinstance(transform, Compose):
                transform.disable_check_args_private()

    def disable_check_args_private(self) -> None:
        self.is_check_args = False
        self.strict = False
        self.main_compose = False

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)

        if not isinstance(force_apply, (bool, int)):
            msg = "force_apply must have bool or int type"
            raise TypeError(msg)

        if self.return_params and self.main_compose:
            data[self.save_key] = OrderedDict()

        need_to_run = force_apply or random.random() < self.p
        if not need_to_run:
            return data

        self.preprocess(data)

        for t in self.transforms:
            data = t(**data)
            data = self.check_data_post_transform(data)

        return self.postprocess(data)

    def run_with_params(self, *, params: dict[int, dict[str, Any]], **data: Any) -> dict[str, Any]:
        """Run transforms with given parameters. Available only for Compose with `return_params=True`."""
        if self._transforms_dict is None:
            raise RuntimeError("`run_with_params` is not available for Compose with `return_params=False`.")

        self.preprocess(data)

        for tr_id, param in params.items():
            tr = self._transforms_dict[tr_id]
            data = tr.apply_with_params(param, **data)
            data = self.check_data_post_transform(data)

        return self.postprocess(data)

    def preprocess(self, data: Any) -> None:
        if self.strict:
            for data_name in data:
                if data_name not in self._available_keys and data_name not in MASK_KEYS:
                    msg = f"Key {data_name} is not in available keys."
                    raise ValueError(msg)
        if self.is_check_args:
            self._check_args(**data)
        if self.main_compose:
            for p in self.processors.values():
                p.ensure_data_valid(data)
            for p in self.processors.values():
                p.preprocess(data)

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.main_compose:
            data = Compose._make_targets_contiguous(data)  # ensure output targets are contiguous
            for p in self.processors.values():
                p.postprocess(data)
        return data

    def to_dict_private(self) -> dict[str, Any]:
        dictionary = super().to_dict_private()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params.to_dict_private() if bbox_processor else None,
                "keypoint_params": (keypoints_processor.params.to_dict_private() if keypoints_processor else None),
                "additional_targets": self.additional_targets,
                "is_check_shapes": self.is_check_shapes,
            },
        )
        return dictionary

    def get_dict_with_id(self) -> dict[str, Any]:
        dictionary = super().get_dict_with_id()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params.to_dict_private() if bbox_processor else None,
                "keypoint_params": (keypoints_processor.params.to_dict_private() if keypoints_processor else None),
                "additional_targets": self.additional_targets,
                "params": None,
                "is_check_shapes": self.is_check_shapes,
            },
        )
        return dictionary

    def _check_args(self, **kwargs: Any) -> None:
        shapes = []

        for data_name, data in kwargs.items():
            internal_data_name = self._additional_targets.get(data_name, data_name)
            if internal_data_name in CHECKED_SINGLE:
                if not isinstance(data, np.ndarray):
                    raise TypeError(f"{data_name} must be numpy array type")
                shapes.append(data.shape[:2])
            if internal_data_name in CHECKED_MULTI and data is not None and len(data):
                if not isinstance(data[0], np.ndarray):
                    raise TypeError(f"{data_name} must be list of numpy arrays")
                shapes.append(data[0].shape[:2])
            if internal_data_name in CHECK_BBOX_PARAM and self.processors.get("bboxes") is None:
                msg = "bbox_params must be specified for bbox transformations"
                raise ValueError(msg)

            if internal_data_name in CHECK_KEYPOINTS_PARAM and self.processors.get("keypoints") is None:
                msg = "keypoints_params must be specified for keypoint transformations"
                raise ValueError(msg)

        if self.is_check_shapes and shapes and shapes.count(shapes[0]) != len(shapes):
            msg = (
                "Height and Width of image, mask or masks should be equal. You can disable shapes check "
                "by setting a parameter is_check_shapes=False of Compose class (do it only if you are sure "
                "about your data consistency)."
            )
            raise ValueError(msg)

    @staticmethod
    def _make_targets_contiguous(data: Any) -> dict[str, Any]:
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                result[key] = np.ascontiguousarray(value)
            else:
                result[key] = value

        return result