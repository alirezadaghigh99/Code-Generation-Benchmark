class FrameData(Mapping[str, Any]):
    """
    A type of the elements returned by indexing the dataset object.
    It can represent both individual frames and batches of thereof;
    in this documentation, the sizes of tensors refer to single frames;
    add the first batch dimension for the collation result.

    Args:
        frame_number: The number of the frame within its sequence.
            0-based continuous integers.
        sequence_name: The unique name of the frame's sequence.
        sequence_category: The object category of the sequence.
        frame_timestamp: The time elapsed since the start of a sequence in sec.
        image_size_hw: The size of the original image in pixels; (height, width)
            tensor of shape (2,). Note that it is optional, e.g. it can be `None`
            if the frame annotation has no size ans image_rgb has not [yet] been
            loaded. Image-less FrameData is valid but mutators like crop/resize
            may fail if the original image size cannot be deduced.
        effective_image_size_hw: The size of the image after mutations such as
            crop/resize in pixels; (height, width). if the image has not been mutated,
            it is equal to `image_size_hw`. Note that it is also optional, for the
            same reason as `image_size_hw`.
        image_path: The qualified path to the loaded image (with dataset_root).
        image_rgb: A Tensor of shape `(3, H, W)` holding the RGB image
            of the frame; elements are floats in [0, 1].
        mask_crop: A binary mask of shape `(1, H, W)` denoting the valid image
            regions. Regions can be invalid (mask_crop[i,j]=0) in case they
            are a result of zero-padding of the image after cropping around
            the object bounding box; elements are floats in {0.0, 1.0}.
        depth_path: The qualified path to the frame's depth map.
        depth_map: A float Tensor of shape `(1, H, W)` holding the depth map
            of the frame; values correspond to distances from the camera;
            use `depth_mask` and `mask_crop` to filter for valid pixels.
        depth_mask: A binary mask of shape `(1, H, W)` denoting pixels of the
            depth map that are valid for evaluation, they have been checked for
            consistency across views; elements are floats in {0.0, 1.0}.
        mask_path: A qualified path to the foreground probability mask.
        fg_probability: A Tensor of `(1, H, W)` denoting the probability of the
            pixels belonging to the captured object; elements are floats
            in [0, 1].
        bbox_xywh: The bounding box tightly enclosing the foreground object in the
            format (x0, y0, width, height). The convention assumes that
            `x0+width` and `y0+height` includes the boundary of the box.
            I.e., to slice out the corresponding crop from an image tensor `I`
            we execute `crop = I[..., y0:y0+height, x0:x0+width]`
        crop_bbox_xywh: The bounding box denoting the boundaries of `image_rgb`
            in the original image coordinates in the format (x0, y0, width, height).
            The convention is the same as for `bbox_xywh`. `crop_bbox_xywh` differs
            from `bbox_xywh` due to padding (which can happen e.g. due to
            setting `JsonIndexDataset.box_crop_context > 0`)
        camera: A PyTorch3D camera object corresponding the frame's viewpoint,
            corrected for cropping if it happened.
        camera_quality_score: The score proportional to the confidence of the
            frame's camera estimation (the higher the more accurate).
        point_cloud_quality_score: The score proportional to the accuracy of the
            frame's sequence point cloud (the higher the more accurate).
        sequence_point_cloud_path: The path to the sequence's point cloud.
        sequence_point_cloud: A PyTorch3D Pointclouds object holding the
            point cloud corresponding to the frame's sequence. When the object
            represents a batch of frames, point clouds may be deduplicated;
            see `sequence_point_cloud_idx`.
        sequence_point_cloud_idx: Integer indices mapping frame indices to the
            corresponding point clouds in `sequence_point_cloud`; to get the
            corresponding point cloud to `image_rgb[i]`, use
            `sequence_point_cloud[sequence_point_cloud_idx[i]]`.
        frame_type: The type of the loaded frame specified in
            `subset_lists_file`, if provided.
        meta: A dict for storing additional frame information.
    """

    frame_number: Optional[torch.LongTensor]
    sequence_name: Union[str, List[str]]
    sequence_category: Union[str, List[str]]
    frame_timestamp: Optional[torch.Tensor] = None
    image_size_hw: Optional[torch.LongTensor] = None
    effective_image_size_hw: Optional[torch.LongTensor] = None
    image_path: Union[str, List[str], None] = None
    image_rgb: Optional[torch.Tensor] = None
    # masks out padding added due to cropping the square bit
    mask_crop: Optional[torch.Tensor] = None
    depth_path: Union[str, List[str], None] = None
    depth_map: Optional[torch.Tensor] = None
    depth_mask: Optional[torch.Tensor] = None
    mask_path: Union[str, List[str], None] = None
    fg_probability: Optional[torch.Tensor] = None
    bbox_xywh: Optional[torch.Tensor] = None
    crop_bbox_xywh: Optional[torch.Tensor] = None
    camera: Optional[PerspectiveCameras] = None
    camera_quality_score: Optional[torch.Tensor] = None
    point_cloud_quality_score: Optional[torch.Tensor] = None
    sequence_point_cloud_path: Union[str, List[str], None] = None
    sequence_point_cloud: Optional[Pointclouds] = None
    sequence_point_cloud_idx: Optional[torch.Tensor] = None
    frame_type: Union[str, List[str], None] = None  # known | unseen
    meta: dict = field(default_factory=lambda: {})

    # NOTE that batching resets this attribute
    _uncropped: bool = field(init=False, default=True)

    def to(self, *args, **kwargs):
        new_params = {}
        for field_name in iter(self):
            value = getattr(self, field_name)
            if isinstance(value, (torch.Tensor, Pointclouds, CamerasBase)):
                new_params[field_name] = value.to(*args, **kwargs)
            else:
                new_params[field_name] = value
        frame_data = type(self)(**new_params)
        frame_data._uncropped = self._uncropped
        return frame_data

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    # the following functions make sure **frame_data can be passed to functions
    def __iter__(self):
        for f in fields(self):
            if f.name.startswith("_"):
                continue

            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return sum(1 for f in iter(self))

    def crop_by_metadata_bbox_(
        self,
        box_crop_context: float,
    ) -> None:
        """Crops the frame data in-place by (possibly expanded) bounding box.
        The bounding box is taken from the object state (usually taken from
        the frame annotation or estimated from the foregroubnd mask).
        If the expanded bounding box does not fit the image, it is clamped,
        i.e. the image is *not* padded.

        Args:
            box_crop_context: rate of expansion for bbox; 0 means no expansion,

        Raises:
            ValueError: If the object does not contain a bounding box (usually when no
                mask annotation is provided)
            ValueError: If the frame data have been cropped or resized, thus the intrinsic
                bounding box is not valid for the current image size.
            ValueError: If the frame does not have an image size (usually a corner case
                when no image has been loaded)
        """
        if self.bbox_xywh is None:
            raise ValueError(
                "Attempted cropping by metadata with empty bounding box. Consider either"
                " to remove_empty_masks or turn off box_crop in the dataset config."
            )

        if not self._uncropped:
            raise ValueError(
                "Trying to apply the metadata bounding box to already cropped "
                "or resized image; coordinates have changed."
            )

        self._crop_by_bbox_(
            box_crop_context,
            self.bbox_xywh,
        )

    def crop_by_given_bbox_(
        self,
        box_crop_context: float,
        bbox_xywh: torch.Tensor,
    ) -> None:
        """Crops the frame data in-place by (possibly expanded) bounding box.
        If the expanded bounding box does not fit the image, it is clamped,
        i.e. the image is *not* padded.

        Args:
            box_crop_context: rate of expansion for bbox; 0 means no expansion,
            bbox_xywh: bounding box in [x0, y0, width, height] format. If float
                tensor, values are floored (after converting to [x0, y0, x1, y1]).

        Raises:
            ValueError: If the frame does not have an image size (usually a corner case
                when no image has been loaded)
        """
        self._crop_by_bbox_(
            box_crop_context,
            bbox_xywh,
        )

    def _crop_by_bbox_(
        self,
        box_crop_context: float,
        bbox_xywh: torch.Tensor,
    ) -> None:
        """Crops the frame data in-place by (possibly expanded) bounding box.
        If the expanded bounding box does not fit the image, it is clamped,
        i.e. the image is *not* padded.

        Args:
            box_crop_context: rate of expansion for bbox; 0 means no expansion,
            bbox_xywh: bounding box in [x0, y0, width, height] format. If float
                tensor, values are floored (after converting to [x0, y0, x1, y1]).

        Raises:
            ValueError: If the frame does not have an image size (usually a corner case
                when no image has been loaded)
        """
        effective_image_size_hw = self.effective_image_size_hw
        if effective_image_size_hw is None:
            raise ValueError("Calling crop on image-less FrameData")

        bbox_xyxy = get_clamp_bbox(
            bbox_xywh,
            image_path=self.image_path,  # pyre-ignore
            box_crop_context=box_crop_context,
        )
        clamp_bbox_xyxy = clamp_box_to_image_bounds_and_round(
            bbox_xyxy,
            image_size_hw=tuple(self.effective_image_size_hw),  # pyre-ignore
        )
        crop_bbox_xywh = bbox_xyxy_to_xywh(clamp_bbox_xyxy)

        if self.fg_probability is not None:
            self.fg_probability = crop_around_box(
                self.fg_probability,
                clamp_bbox_xyxy,
                self.mask_path,  # pyre-ignore
            )
        if self.image_rgb is not None:
            self.image_rgb = crop_around_box(
                self.image_rgb,
                clamp_bbox_xyxy,
                self.image_path,  # pyre-ignore
            )

        depth_map = self.depth_map
        if depth_map is not None:
            clamp_bbox_xyxy_depth = rescale_bbox(
                clamp_bbox_xyxy, tuple(depth_map.shape[-2:]), effective_image_size_hw
            ).long()
            self.depth_map = crop_around_box(
                depth_map,
                clamp_bbox_xyxy_depth,
                self.depth_path,  # pyre-ignore
            )

        depth_mask = self.depth_mask
        if depth_mask is not None:
            clamp_bbox_xyxy_depth = rescale_bbox(
                clamp_bbox_xyxy, tuple(depth_mask.shape[-2:]), effective_image_size_hw
            ).long()
            self.depth_mask = crop_around_box(
                depth_mask,
                clamp_bbox_xyxy_depth,
                self.mask_path,  # pyre-ignore
            )

        # changing principal_point according to bbox_crop
        if self.camera is not None:
            adjust_camera_to_bbox_crop_(
                camera=self.camera,
                image_size_wh=effective_image_size_hw.flip(dims=[-1]),
                clamp_bbox_xywh=crop_bbox_xywh,
            )

        # pyre-ignore
        self.effective_image_size_hw = crop_bbox_xywh[..., 2:].flip(dims=[-1])
        self._uncropped = False

    def resize_frame_(self, new_size_hw: torch.LongTensor) -> None:
        """Resizes frame data in-place according to given dimensions.

        Args:
            new_size_hw: target image size [height, width], a LongTensor of shape (2,)

        Raises:
            ValueError: If the frame does not have an image size (usually a corner case
                when no image has been loaded)
        """

        effective_image_size_hw = self.effective_image_size_hw
        if effective_image_size_hw is None:
            raise ValueError("Calling resize on image-less FrameData")

        image_height, image_width = new_size_hw.tolist()

        if self.fg_probability is not None:
            self.fg_probability, _, _ = resize_image(
                self.fg_probability,
                image_height=image_height,
                image_width=image_width,
                mode="nearest",
            )

        if self.image_rgb is not None:
            self.image_rgb, _, self.mask_crop = resize_image(
                self.image_rgb, image_height=image_height, image_width=image_width
            )

        if self.depth_map is not None:
            self.depth_map, _, _ = resize_image(
                self.depth_map,
                image_height=image_height,
                image_width=image_width,
                mode="nearest",
            )

        if self.depth_mask is not None:
            self.depth_mask, _, _ = resize_image(
                self.depth_mask,
                image_height=image_height,
                image_width=image_width,
                mode="nearest",
            )

        if self.camera is not None:
            if self.image_size_hw is None:
                raise ValueError(
                    "image_size_hw has to be defined for resizing FrameData with cameras."
                )
            adjust_camera_to_image_scale_(
                camera=self.camera,
                original_size_wh=effective_image_size_hw.flip(dims=[-1]),
                new_size_wh=new_size_hw.flip(dims=[-1]),  # pyre-ignore
            )

        self.effective_image_size_hw = new_size_hw
        self._uncropped = False

    @classmethod
    def collate(cls, batch):
        """
        Given a list objects `batch` of class `cls`, collates them into a batched
        representation suitable for processing with deep networks.
        """

        elem = batch[0]

        if isinstance(elem, cls):
            pointcloud_ids = [id(el.sequence_point_cloud) for el in batch]
            id_to_idx = defaultdict(list)
            for i, pc_id in enumerate(pointcloud_ids):
                id_to_idx[pc_id].append(i)

            sequence_point_cloud = []
            sequence_point_cloud_idx = -np.ones((len(batch),))
            for i, ind in enumerate(id_to_idx.values()):
                sequence_point_cloud_idx[ind] = i
                sequence_point_cloud.append(batch[ind[0]].sequence_point_cloud)
            assert (sequence_point_cloud_idx >= 0).all()

            override_fields = {
                "sequence_point_cloud": sequence_point_cloud,
                "sequence_point_cloud_idx": sequence_point_cloud_idx.tolist(),
            }
            # note that the pre-collate value of sequence_point_cloud_idx is unused

            collated = {}
            for f in fields(elem):
                if not f.init:
                    continue

                list_values = override_fields.get(
                    f.name, [getattr(d, f.name) for d in batch]
                )
                collated[f.name] = (
                    cls.collate(list_values)
                    if all(list_value is not None for list_value in list_values)
                    else None
                )
            return cls(**collated)

        elif isinstance(elem, Pointclouds):
            return join_pointclouds_as_batch(batch)

        elif isinstance(elem, CamerasBase):
            # TODO: don't store K; enforce working in NDC space
            return join_cameras_as_batch(batch)
        else:
            return torch.utils.data.dataloader.default_collate(batch)

