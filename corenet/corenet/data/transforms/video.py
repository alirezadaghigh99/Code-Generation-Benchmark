class CropByBoundingBox(BaseTransformation):
    """Crops video frames based on bounding boxes and adjusts the @targets
    "box_coordinates" annotations.
    Before cropping, the bounding boxes are expanded with @multiplier, while the
    "box_coordinates" cover the original areas of the image.
    Note that the cropped images may be padded with 0 values in the boundaries of the
    cropped image when the bounding boxes are near the edges.

    Frames with invalid bounding boxes (with x0=y0=x1=y1=-1, or with area <5) will be
    blacked out in the output. Alternatively, we could have dropped them, which is not
    implemented yet.
    """

    BBOX_MIN_AREA = 5  # Minimum valid bounding box area (in pixels).

    def __init__(
        self,
        opts: argparse.Namespace,
        image_size: Optional[Tuple[int, int]] = None,
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.is_training = is_training
        self.multiplier = getattr(
            opts, "video_augmentation.crop_by_bounding_box.multiplier"
        )
        self.multiplier_range = getattr(
            opts, "video_augmentation.crop_by_bounding_box.multiplier_range"
        )
        if image_size is None:
            self.image_size = getattr(
                opts, "video_augmentation.crop_by_bounding_box.image_size"
            )
        else:
            self.image_size = image_size
            assert image_size is not None, (
                "Please provide --video-augmentation.crop-by-bounding-box.image_size"
                " argument."
            )
        self.channel_first = getattr(
            opts, "video_augmentation.crop_by_bounding_box.channel_first"
        )
        self.interpolation = getattr(
            opts, "video_augmentation.crop_by_bounding_box.interpolation"
        )

    def __call__(self, data: Dict, *args, **kwargs) -> Dict:
        """
        Tensor shape abbreviations:
            N: Number of clips.
            T, T_audio, T_video: Temporal lengths.
            C: Number of color channels.
            H, W: Height, Width.

        Args:
            data: mapping of: {
                "samples": {
                    "video": Tensor of shape: [N, C, T, H, W] if self.channel_first else [N, T, C, H, W]
                },
                "targets": {
                    "traces": {
                        "<object_trace_uuid>": {
                            "box_coordinates": FloatTensor[N, T, 4],  # x0, y0, x1, y1
                        }
                    },
                    "labels": IntTensor[N, T],
                }
            }

        Note:
            This transformation does not modify the "labels". If frames that are
            blacked out due to having invalid bounding boxes need a different label,
            datasets should alter the labels according to the following logic:
            ```
                data = CropByBoundingBox(opts)(data)
                trace, = data["targets"]["traces"].values()
                is_blacked_out = torch.all(trace["box_coordinates"] == -1, dim=2)
                data["targets"]["labels"][is_blacked_out] = <custom_label>
            ```
        """

        traces = data["targets"]["traces"]
        trace_identity = random.choice(list(traces.keys()))
        trace = traces[trace_identity]
        video = data["samples"]["video"]
        if self.channel_first:
            video = video.movedim(2, 1)

        N, T, C, H, W = video.shape
        expected_box_coordinates_shape = (N, T, 4)

        box_coordinates = trace["box_coordinates"]
        assert box_coordinates.shape == expected_box_coordinates_shape, (
            f"Unexpected shape {trace['box_coordinates'].shape} !="
            f" {expected_box_coordinates_shape}"
        )
        if self.is_training and self.multiplier_range is not None:
            multiplier = random.uniform(*self.multiplier_range)
        else:
            multiplier = self.multiplier

        expanded_corners, box_coordinates = self.expand_boxes(
            trace["box_coordinates"], multiplier, height=H, width=W
        )  # (NxTx4, NxTx4)

        expanded_corners = (
            (expanded_corners * torch.tensor([W, H, W, H]).float()).round().int()
        )  # NxTx4

        result = torch.empty(
            [N * T, C, *self.image_size],
            dtype=video.dtype,
            device=video.device,
        )
        for images, crop_corners, result_placeholder in zip(
            video.reshape(-1, C, H, W), expanded_corners.reshape(-1, 4).tolist(), result
        ):
            # TODO: add video_augmentation.crop_by_bounding_box.antialias argument to
            # experiment on antialias parameter of torchvision's resize function.
            width = crop_corners[2] - crop_corners[0]
            height = crop_corners[3] - crop_corners[1]
            if (
                width * height < CropByBoundingBox.BBOX_MIN_AREA
                or width < 0
                or height < 0
            ):
                # If the bounding box is invalid or too small, avoid cropping.
                result_placeholder[...] = 0.0  # Create black frames
            else:
                result_placeholder[...] = FV.resized_crop(
                    images,
                    left=crop_corners[0],
                    top=crop_corners[1],
                    width=width,
                    height=height,
                    size=self.image_size,
                    interpolation=InterpolationMode[self.interpolation.upper()],
                    antialias=True,
                )
        data["samples"]["video"] = result.reshape(N, T, C, *self.image_size)
        data["targets"]["traces"] = {
            trace_identity: {**trace, "box_coordinates": box_coordinates}
        }
        return data

    def expand_boxes(
        self, box_coordinates: torch.Tensor, multiplier: float, width: int, height: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            box_coordinates: Tensor of shape [..., 4] with (x0, y0, x1, y1) in [0,1].
            multiplier: The multiplier to expand the bounding box coordinates.

        Outputs (tuple items):
            expanded_corners: Tensor of shape [..., 4] with (x0, y0, x1, y1), containing
                the coordinates for cropping. Because of the expansion, coordinates
                could be negative or >1.
            box_coordinates: Tensor of shape [..., 4] with (x0, y0, x1, y1) in [0,1] to
                be used as bounding boxes after cropping.
            height: Height of the frame (in pixels).
            width: Width of the frame (in pixels).
        """
        x0 = box_coordinates[..., 0]  # Shape: NxT
        y0 = box_coordinates[..., 1]
        x1 = box_coordinates[..., 2]
        y1 = box_coordinates[..., 3]
        area = (x1 - x0) * width * (y1 - y0) * height
        invisible_mask = area < CropByBoundingBox.BBOX_MIN_AREA

        dw = (x1 - x0) * (multiplier - 1) / 2
        dh = (y1 - y0) * (multiplier - 1) / 2
        expanded_corners = torch.stack(
            [
                x0 - dw,
                y0 - dh,
                x1 + dw,
                y1 + dh,
            ],
            dim=-1,
        )

        # If multiplier is 1, new box_coordinates should cover the whole image (i.e.
        # [0., 0., 1., 1.]), as image was cropped based on the box_coordinates. For
        # multiplier > 1, new box_coordinates should have a small margin within the
        # boundaries (i.e. [new_x0, new_y0, 1-new_x0, 1-new_y0]).
        box_coordinates = torch.empty_like(box_coordinates)
        box_coordinates[..., :2] = self.get_new_x0(multiplier)
        box_coordinates[..., 2:] = 1 - box_coordinates[..., :2]
        expanded_corners[invisible_mask] = -1
        box_coordinates[invisible_mask] = -1
        return expanded_corners, box_coordinates

    @classmethod
    def get_new_x0(cls, multiplier: float) -> float:
        # new_width = old_width * multiplier
        # new_x0 = [(new_width - old_width) / 2] / new_width
        # => new_x0 = (1 - 1/multiplier) / 2
        return (1 - 1 / multiplier) / 2

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.enable",
            action="store_true",
            help=(
                "Use {}. This flag is useful when you want to study the effect of"
                " different transforms. Default to False.".format(cls.__name__)
            ),
        )
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.image-size",
            type=JsonValidator(Tuple[int, int]),
            default=None,
            help=(
                "Sizes [height, width] of the video frames after cropping. Defaults to"
                " None"
            ),
        )
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.channel-first",
            action="store_true",
            default=False,
            help=(
                "If true, the video shape is [N, C, T, H, W]. Otherwise:"
                " [N, T, C, H, W]. Defaults to False."
            ),
        ),
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.multiplier-range",
            type=float,
            nargs=2,
            default=None,
            help=(
                "The bounding boxes get randomly expanded within the range before"
                " cropping. Useful for zooming in/out. Default None means no expansion"
                " of the bounding box."
            ),
        )
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.multiplier",
            type=float,
            default=1,
            help=(
                "The bounding boxes get expanded by this multiplier before cropping."
                " Useful for zooming in/out. Defaults to 1."
            ),
        )
        group.add_argument(
            "--video-augmentation.crop-by-bounding-box.interpolation",
            type=str,
            default="bilinear",
            choices=SUPPORTED_PYTORCH_INTERPOLATIONS,
            help="Desired interpolation method. Defaults to bilinear.",
        )

        return parser

    def __repr__(self) -> str:
        return "{}(image size={}, channel_first={}, multiplier={})".format(
            self.__class__.__name__,
            self.image_size,
            self.channel_first,
            self.multiplier,
        )

class ShuffleAudios(BaseTransformation):
    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool,
        is_evaluation: bool,
        item_index: int,
        *args,
        **kwargs,
    ) -> None:
        """Transforms a batch of audio-visual clips. Generates binary labels, useful for
        self-supervised audio-visual training.

        At each invocation, a subset of clips within video (batch) get their audios
        shuffled. The ratio of clips that participate in the shuffling is configurable
        by argparse options.

        When training, the shuffle order is random. When evaluating, the shuffle order
        is deterministic.

        Args:
            is_training: When False, decide to shuffle the audios or not
                deterministically.
            is_evaluation: Combined with @is_training, determines which shuffle ratio
                argument to use (train/val/eval).
            item_index: Used for deterministic shuffling based on the item_index.
        """
        super().__init__(opts, *args, **kwargs)
        self.item_index = item_index
        self.is_training = is_training
        if is_training:
            self.shuffle_ratio = getattr(
                opts, "video_augmentation.shuffle_audios.shuffle_ratio_train"
            )
        elif is_evaluation:
            self.shuffle_ratio = getattr(
                opts, "video_augmentation.shuffle_audios.shuffle_ratio_test"
            )
        else:
            self.shuffle_ratio = getattr(
                opts, "video_augmentation.shuffle_audios.shuffle_ratio_val"
            )
        self.generate_frame_level_targets = getattr(
            opts,
            "video_augmentation.shuffle_audios.generate_frame_level_targets",
        )
        self.target_key = getattr(opts, "video_augmentation.shuffle_audios.target_key")
        self.debug_mode = getattr(opts, "video_augmentation.shuffle_audios.debug_mode")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--video-augmentation.shuffle-audios.shuffle-ratio-train",
            type=float,
            default=0.5,
            help=(
                "Ratio of training videos with shuffled audio samples. Defaults to 0.5."
            ),
        )
        group.add_argument(
            "--video-augmentation.shuffle-audios.shuffle-ratio-val",
            type=float,
            default=0.5,
            help=(
                "Ratio of validation videos with shuffled audio samples. Defaults to "
                " 0.5."
            ),
        )
        group.add_argument(
            "--video-augmentation.shuffle-audios.shuffle-ratio-test",
            type=float,
            default=0.5,
            help="Ratio of test videos with shuffled audio samples. Defaults to 0.5.",
        )
        group.add_argument(
            "--video-augmentation.shuffle-audios.generate-frame-level-targets",
            default=False,
            action="store_true",
            help=(
                "If true, the generated targets will be 2-dimensional (n_clips x "
                "n_frames). Otherwise, targets will be 1 dimensional (n_clips)."
                " Defaults to False."
            ),
        )
        group.add_argument(
            "--video-augmentation.shuffle-audios.target-key",
            default="is_shuffled",
            type=str,
            help=(
                "Defaults to 'is_shuffled'. Name of the sub-key in data['targets'] "
                " to store the labels tensor. For each clip index `i`, we will have"
                " data['targets']['is_shuffled'][i] == 0 iff audio of the clip matches"
                " the video, otherwise 1."
            ),
        )
        group.add_argument(
            "--video-augmentation.shuffle-audios.debug-mode",
            default=False,
            action="store_true",
            help=(
                "If enabled, the permutation used for shuffling the clip audios will be"
                " added to data['samples']['metadata']['shuffled_audio_permutation']"
                " for debugging purposes. Defaults to False."
            ),
        )

        return parser

    @staticmethod
    def _single_cycle_permutation(
        numel: int, is_training: bool, device: torch.device
    ) -> torch.LongTensor:
        """
        Returns a permutation of values 0 to @numel-1 that has the following property:
        For each index 0 <= i < numel: result[i] != i.

        Args:
            numel: Number of elements in the output permutation (must be >1).
            is_training: If true, the output permutation will be deterministic.
            device: Torch device (e.g. cuda, cpu) to use for output tensor.
        """
        assert numel > 1, "Cannot create a single-cycle permutation with <= 1 elements."

        deterministic_single_cycle_perm = torch.roll(
            torch.arange(numel, device=device), numel // 2
        )
        if not is_training:
            return deterministic_single_cycle_perm

        random_perm = torch.randperm(numel, device=device)

        random_perm_inv = torch.empty_like(random_perm)
        random_perm_inv[random_perm] = torch.arange(numel, device=device)

        # Proof that this implementation satisfies output[i] != i criteria:
        # 1. We know deterministic_single_cycle_perm[i] != i, because of the way it is
        #    constructed ([n//2, n//2+1, ..., n, 1, 2, ..., n//2-1]).
        # 2. ``rand_perm`` is a non-deterministic random permutation, and
        #    ``rand_perm_inv`` is the inverse of `rand_perm`. That means for each
        #    0 <= i < numel, we have: rand_perm_inv[rand_perm[i]] == i.
        # 3. Proof by contradiction: Let's assume, for 0 <= i < numel, i == output[i]:
        #    Thus: random_perm[deterministic_single_cycle_perm[random_perm_inv]][i] == i
        # 4. For any two torch tensors a, b that expression `a[b]`` is valid, we have
        #    a[b][i] == a[b[i]]. Thus, we can rewrite the assumption of step 3 as:
        #    i == random_perm[deterministic_single_cycle_perm[random_perm_inv[i]]]
        # 5. Now, apply rand_perm_inv[] on both sides of the equality:
        #    rand_perm_inv[i] == deterministic_single_cycle_perm[random_perm_inv[i]]
        #    Then, alias rand_perm_inv[i] as x. Then we will have:
        #    x == deterministic_single_cycle_perm[x]
        # 6. Assumption of step (3) leads to (5) which contradicts (1). Thus, assumption
        #    of step (3) is false. Thus, output[i] != i
        return random_perm[deterministic_single_cycle_perm[random_perm_inv]]

    def _random_outcome(self, n: int) -> torch.Tensor:
        """Returns a pseudo random tensor of size n in range [0, 1]. For evaluation,
        the outcome is a deterministic function of n and `self.item_index`

        Args:
            n: Length of the output tensor.

        Returns: A tensor of length n, of random floats uniformly distributed between
            0-1. The output is deterministic iff self.is_training is False.
        """
        if self.is_training:
            return torch.rand(n)
        else:
            return (
                (((self.item_index + 1) % torch.pi) * (torch.arange(n) + 1)) % torch.pi
            ) / torch.pi

    def _random_participants_mask(self, n: int) -> torch.BoolTensor:
        """Returns a pseudo random boolean tensor of size n, where exactly ``int(
        self.shuffle_ratio * n)`` indices are True, and the rest are False.
        """
        x = self._random_outcome(n)
        x = x.argsort() < self.shuffle_ratio * n - 1e-8
        return x

    def __call__(self, data: Dict) -> Dict:
        audio = data["samples"]["audio"]
        N = len(audio)
        if N == 1:
            shuffled_permutation = torch.tensor([0], device=audio.device)
            is_shuffling_participant_mask = torch.tensor([False], device=audio.device)
        elif N > 1:
            shuffled_permutation = self._single_cycle_permutation(
                N, device=audio.device, is_training=self.is_training
            )
            is_shuffling_participant_mask = self._random_participants_mask(N)
            shuffled_permutation = torch.where(
                is_shuffling_participant_mask,
                shuffled_permutation,
                torch.arange(N),
            )
        else:
            raise ValueError("Insufficient clips (N={N}) in batch.")

        data["samples"]["audio"] = audio[shuffled_permutation]
        if self.debug_mode:
            data["samples"]["metadata"][
                "shuffled_audio_permutation"
            ] = shuffled_permutation

        target_dims = 2 if self.generate_frame_level_targets else 1
        labels = torch.zeros(
            data["samples"]["video"].shape[:target_dims],
            device=audio.device,
            dtype=torch.long,
        )
        labels[is_shuffling_participant_mask] = 1.0  # 1 means shuffled
        data["targets"][self.target_key] = labels
        return data

