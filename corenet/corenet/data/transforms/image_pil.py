class ToTensor(BaseTransformation):
    """
    This method converts an image into a tensor and optionally normalizes by a mean and std.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        img_dtype = getattr(opts, "image_augmentation.to_tensor.dtype", "float")
        mean_std_normalization_enable = getattr(
            opts, "image_augmentation.to_tensor.mean_std_normalization.enable"
        )
        normalization_mean = getattr(
            opts, "image_augmentation.to_tensor.mean_std_normalization.mean"
        )
        normalization_std = getattr(
            opts, "image_augmentation.to_tensor.mean_std_normalization.std"
        )

        if mean_std_normalization_enable:
            assert (
                normalization_mean is not None
            ), "--image_augmentation.to_tensor.mean_std_normalization.mean must be specified when --image_augmentation.to_tensor.mean_std_normalization.enable is set to true."
            assert (
                normalization_std is not None
            ), "--image_augmentation.to_tensor.mean_std_normalization.std must be specified when --image_augmentation.to_tensor.mean_std_normalization.enable is set to true."

            if isinstance(normalization_mean, list):
                assert (
                    len(normalization_mean) == 3
                ), "--image_augmentation.to_tensor.mean_std_normalization.mean must be a list of length 3 or a scalar."

            if isinstance(normalization_std, list):
                assert (
                    len(normalization_std) == 3
                ), "--image_augmentation.to_tensor.mean_std_normalization.std must be a list of length 3 or a scalar."

        self.img_dtype = torch.float
        self.norm_factor = 255
        if img_dtype in ["half", "float16"]:
            self.img_dtype = torch.float16
        elif img_dtype in ["uint8"]:
            self.img_dtype = torch.uint8
            self.norm_factor = 1

        self.mean_std_normalization_enable = mean_std_normalization_enable
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--image-augmentation.to-tensor.dtype",
            type=str,
            default="float",
            help="Tensor data type. Default is float",
        )
        parser.add_argument(
            "--image-augmentation.to-tensor.mean-std-normalization.enable",
            action="store_true",
            default=False,
            help="This flag is used to normalize a tensor by a dataset's mean and std. Defaults to False.",
        )
        parser.add_argument(
            "--image-augmentation.to-tensor.mean-std-normalization.mean",
            type=float,
            nargs="+",
            default=None,
            help="The mean used to normalize the input. Defaults to None.",
        )
        parser.add_argument(
            "--image-augmentation.to-tensor.mean-std-normalization.std",
            type=float,
            nargs="+",
            default=None,
            help="The standard deviation used to normalize the input. Defaults to None.",
        )
        return parser

    def __repr__(self):
        if self.mean_std_normalization_enable:
            return "{}(dtype={}, norm_factor={}, mean_std_normalization_enable={}, normalization_mean={}, normalization_std={})".format(
                self.__class__.__name__,
                self.img_dtype,
                self.norm_factor,
                self.mean_std_normalization_enable,
                self.normalization_mean,
                self.normalization_std,
            )
        else:
            return "{}(dtype={}, norm_factor={})".format(
                self.__class__.__name__,
                self.img_dtype,
                self.norm_factor,
            )

    def __call__(self, data: Dict) -> Dict:
        # HWC --> CHW
        img = data["image"]

        if F._is_pil_image(img):
            # convert PIL image to tensor
            img = F.pil_to_tensor(img).contiguous()

        data["image"] = img.to(dtype=self.img_dtype).div(self.norm_factor)

        if self.mean_std_normalization_enable:
            data["image"] = F.normalize(
                data["image"],
                mean=self.normalization_mean,
                std=self.normalization_std,
            )

        if "mask" in data:
            mask = data.pop("mask")
            mask = np.array(mask)

            if len(mask.shape) not in (2, 3):
                logger.error(
                    "Mask needs to be 2- or 3-dimensional. Got: {}".format(mask.shape)
                )
            data["mask"] = torch.as_tensor(mask, dtype=torch.long)

        if "box_coordinates" in data:
            boxes = data.pop("box_coordinates")
            data["box_coordinates"] = torch.as_tensor(boxes, dtype=torch.float)

        if "box_labels" in data:
            box_labels = data.pop("box_labels")
            data["box_labels"] = torch.as_tensor(box_labels)

        if "instance_mask" in data:
            assert "instance_coords" in data
            instance_masks = data.pop("instance_mask")
            data["instance_mask"] = instance_masks.to(dtype=torch.long)

            instance_coords = data.pop("instance_coords")
            data["instance_coords"] = torch.as_tensor(
                instance_coords, dtype=torch.float
            )
        return data

class RandomResizedCrop(BaseTransformation, T.RandomResizedCrop):
    """
    This class crops a random portion of an image and resize it to a given size.
    """

    def __init__(
        self, opts: argparse.Namespace, size: Union[Sequence, int], *args, **kwargs
    ) -> None:
        interpolation = getattr(
            opts, "image_augmentation.random_resized_crop.interpolation"
        )
        scale = getattr(opts, "image_augmentation.random_resized_crop.scale")
        ratio = getattr(opts, "image_augmentation.random_resized_crop.aspect_ratio")

        BaseTransformation.__init__(self, opts=opts)

        T.RandomResizedCrop.__init__(
            self,
            size=size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.random-resized-crop.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-resized-crop.interpolation",
            type=str,
            default="bilinear",
            choices=list(INTERPOLATION_MODE_MAP.keys()),
            help="Interpolation method for resizing. Defaults to bilinear.",
        )
        group.add_argument(
            "--image-augmentation.random-resized-crop.scale",
            type=JsonValidator(Tuple[float, float]),
            default=(0.08, 1.0),
            help="Specifies the lower and upper bounds for the random area of the crop, before resizing."
            " The scale is defined with respect to the area of the original image. Defaults to "
            "(0.08, 1.0)",
        )
        group.add_argument(
            "--image-augmentation.random-resized-crop.aspect-ratio",
            type=float or tuple,
            default=(3.0 / 4.0, 4.0 / 3.0),
            help="lower and upper bounds for the random aspect ratio of the crop, before resizing. "
            "Defaults to (3./4., 4./3.)",
        )
        return parser

    def get_rrc_params(self, image: Image.Image) -> Tuple[int, int, int, int]:
        return T.RandomResizedCrop.get_params(
            img=image, scale=self.scale, ratio=self.ratio
        )

    def __call__(self, data: Dict) -> Dict:
        """
        Input data format:
            data: mapping of: {
                "image": [Height, Width, Channels],
                "mask": [Height, Width],
                "box_coordinates": [Num_boxes, x, y, w, h],
                "box_labels: : [Num_boxes],
            }
        Output data format: Same as the input
        """
        img = data["image"]
        i, j, h, w = self.get_rrc_params(image=img)
        data = _crop_fn(data=data, top=i, left=j, height=h, width=w)
        return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)

    def __repr__(self) -> str:
        return "{}(scale={}, ratio={}, size={}, interpolation={})".format(
            self.__class__.__name__,
            self.scale,
            self.ratio,
            self.size,
            self.interpolation,
        )

