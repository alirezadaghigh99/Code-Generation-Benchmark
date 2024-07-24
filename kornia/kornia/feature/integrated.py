class LightGlueMatcher(GeometryAwareDescriptorMatcher):
    """LightGlue-based matcher in kornia API.

    This is based on the original code from paper "LightGlue: Local Feature Matching at Light Speed".
    See :cite:`LightGlue2023` for more details.

    Args:
        feature_name: type of feature for matching, can be `disk` or `superpoint`.
        params: LightGlue params.
    """

    known_modes: ClassVar[List[str]] = [
        "aliked",
        "dedodeb",
        "dedodeg",
        "disk",
        "dog_affnet_hardnet",
        "doghardnet",
        "keynet_affnet_hardnet",
        "sift",
        "superpoint",
    ]

    def __init__(self, feature_name: str = "disk", params: Dict = {}) -> None:  # type: ignore
        feature_name_: str = feature_name.lower()
        super().__init__(feature_name_)
        self.feature_name = feature_name_
        self.params = params
        self.matcher = LightGlue(self.feature_name, **params)

    def forward(
        self,
        desc1: Tensor,
        desc2: Tensor,
        lafs1: Tensor,
        lafs2: Tensor,
        hw1: Optional[Tuple[int, int]] = None,
        hw2: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
            lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
            lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.

        Return:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.
        """
        if (desc1.shape[0] < 2) or (desc2.shape[0] < 2):
            return _no_match(desc1)
        keypoints1 = get_laf_center(lafs1)
        keypoints2 = get_laf_center(lafs2)
        if len(desc1.shape) == 2:
            desc1 = desc1.unsqueeze(0)
        if len(desc2.shape) == 2:
            desc2 = desc2.unsqueeze(0)
        dev = lafs1.device
        if hw1 is None:
            hw1_ = keypoints1.max(dim=1)[0].squeeze().flip(0)
        else:
            hw1_ = torch.tensor(hw1, device=dev)
        if hw2 is None:
            hw2_ = keypoints2.max(dim=1)[0].squeeze().flip(0)
        else:
            hw2_ = torch.tensor(hw2, device=dev)
        ori0 = deg2rad(get_laf_orientation(lafs1).reshape(1, -1))
        ori0[ori0 < 0] += 2.0 * pi
        ori1 = deg2rad(get_laf_orientation(lafs2).reshape(1, -1))
        ori1[ori1 < 0] += 2.0 * pi
        input_dict = {
            "image0": {
                "keypoints": keypoints1,
                "scales": get_laf_scale(lafs1).reshape(1, -1),
                "oris": ori0,
                "lafs": lafs1,
                "descriptors": desc1,
                "image_size": hw1_.flip(0).reshape(-1, 2).to(dev),
            },
            "image1": {
                "keypoints": keypoints2,
                "lafs": lafs2,
                "scales": get_laf_scale(lafs2).reshape(1, -1),
                "oris": ori1,
                "descriptors": desc2,
                "image_size": hw2_.flip(0).reshape(-1, 2).to(dev),
            },
        }
        pred = self.matcher(input_dict)
        matches0, mscores0 = pred["matches0"], pred["matching_scores0"]
        valid = matches0 > -1
        matches = torch.stack([torch.where(valid)[1], matches0[valid]], -1)
        return mscores0[valid].reshape(-1, 1), matches

class LAFDescriptor(Module):
    r"""Module to get local descriptors, corresponding to LAFs (keypoints).

    Internally uses :func:`~kornia.feature.get_laf_descriptors`.

    Args:
        patch_descriptor_module: patch descriptor module, e.g. :class:`~kornia.feature.SIFTDescriptor`
            or :class:`~kornia.feature.HardNet`. Default: :class:`~kornia.feature.HardNet`.
        patch_size: patch size in pixels, which descriptor expects.
        grayscale_descriptor: ``True`` if patch_descriptor expects single-channel image.
    """

    def __init__(
        self, patch_descriptor_module: Optional[Module] = None, patch_size: int = 32, grayscale_descriptor: bool = True
    ) -> None:
        super().__init__()
        if patch_descriptor_module is None:
            patch_descriptor_module = HardNet(True)
        self.descriptor = patch_descriptor_module
        self.patch_size = patch_size
        self.grayscale_descriptor = grayscale_descriptor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(descriptor={self.descriptor.__repr__()}, "
            f"patch_size={self.patch_size}, "
            f"grayscale_descriptor='{self.grayscale_descriptor})"
        )

    def forward(self, img: Tensor, lafs: Tensor) -> Tensor:
        r"""Three stage local feature detection.

        First the location and scale of interest points are determined by
        detect function. Then affine shape and orientation.

        Args:
            img: image features with shape :math:`(B,C,H,W)`.
            lafs: local affine frames :math:`(B,N,2,3)`.

        Returns:
            Local descriptors of shape :math:`(B,N,D)` where :math:`D` is descriptor size.
        """
        return get_laf_descriptors(img, lafs, self.descriptor, self.patch_size, self.grayscale_descriptor)

