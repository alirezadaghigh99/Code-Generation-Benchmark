class DualTransform(BasicTransform):
    """A base class for transformations that should be applied both to an image and its corresponding properties
    such as masks, bounding boxes, and keypoints. This class ensures that when a transform is applied to an image,
    all associated entities are transformed accordingly to maintain consistency between the image and its annotations.

    Properties:
        targets (dict[str, Callable[..., Any]]): Defines the types of targets (e.g., image, mask, bboxes, keypoints)
            that the transform should be applied to and maps them to the corresponding methods.

    Methods:
        apply_to_bbox(bbox: BoxInternalType, *args: Any, **params: Any) -> BoxInternalType:
            Applies the transform to a single bounding box. Should be implemented in the subclass.

        apply_to_keypoint(keypoint: KeypointInternalType, *args: Any, **params: Any) -> KeypointInternalType:
            Applies the transform to a single keypoint. Should be implemented in the subclass.

        apply_to_bboxes(bboxes: Sequence[BoxType], *args: Any, **params: Any) -> Sequence[BoxType]:
            Applies the transform to a list of bounding boxes. Delegates to `apply_to_bbox` for each bounding box.

        apply_to_keypoints(keypoints: Sequence[KeypointType], *args: Any, **params: Any) -> Sequence[KeypointType]:
            Applies the transform to a list of keypoints. Delegates to `apply_to_keypoint` for each keypoint.

        apply_to_mask(mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
            Applies the transform specifically to a single mask.

        apply_to_masks(masks: Sequence[np.ndarray], **params: Any) -> list[np.ndarray]:
            Applies the transform to a list of masks. Delegates to `apply_to_mask` for each mask.

    Note:
        This class is intended to be subclassed and should not be used directly. Subclasses are expected to
        implement the specific logic for each type of target (e.g., image, mask, bboxes, keypoints) in the
        corresponding `apply_to_*` methods.

    """

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }

    def apply_to_bbox(self, bbox: BoxInternalType, *args: Any, **params: Any) -> BoxInternalType:
        msg = f"Method apply_to_bbox is not implemented in class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, *args: Any, **params: Any) -> KeypointInternalType:
        msg = f"Method apply_to_keypoint is not implemented in class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def apply_to_global_label(self, label: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        msg = f"Method apply_to_global_label is not implemented in class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def apply_to_bboxes(self, bboxes: Sequence[BoxType], *args: Any, **params: Any) -> Sequence[BoxType]:
        return [
            self.apply_to_bbox(cast(BoxInternalType, tuple(cast(BoxInternalType, bbox[:4]))), **params)
            + tuple(bbox[4:])
            for bbox in bboxes
        ]

    def apply_to_keypoints(
        self,
        keypoints: Sequence[KeypointType],
        *args: Any,
        **params: Any,
    ) -> Sequence[KeypointType]:
        return [
            self.apply_to_keypoint(cast(KeypointInternalType, tuple(keypoint[:4])), **params) + tuple(keypoint[4:])
            for keypoint in keypoints
        ]

    def apply_to_mask(self, mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(mask, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_masks(self, masks: Sequence[np.ndarray], **params: Any) -> list[np.ndarray]:
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def apply_to_global_labels(self, labels: Sequence[np.ndarray], **params: Any) -> list[np.ndarray]:
        return [self.apply_to_global_label(label, **params) for label in labels]

class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    _targets = Targets.IMAGE

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {"image": self.apply}

class NoOp(DualTransform):
    """Identity transform (does nothing).

    Targets:
        image, mask, bboxes, keypoints, global_label
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS, Targets.GLOBAL_LABEL)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        return keypoint

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return bbox

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return img

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        return mask

    def apply_to_global_label(self, label: np.ndarray, **params: Any) -> np.ndarray:
        return label

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()

class NoOp(DualTransform):
    """Identity transform (does nothing).

    Targets:
        image, mask, bboxes, keypoints, global_label
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS, Targets.GLOBAL_LABEL)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        return keypoint

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return bbox

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return img

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        return mask

    def apply_to_global_label(self, label: np.ndarray, **params: Any) -> np.ndarray:
        return label

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()

