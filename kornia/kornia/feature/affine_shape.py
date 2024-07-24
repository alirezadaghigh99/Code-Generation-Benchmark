class LAFAffineShapeEstimator(nn.Module):
    """Module, which extracts patches using input images and local affine frames (LAFs).

    Then runs :class:`~kornia.feature.PatchAffineShapeEstimator` on patches to estimate LAFs shape.

    Then original LAF shape is replaced with estimated one. The original LAF orientation is not preserved,
    so it is recommended to first run LAFAffineShapeEstimator and then LAFOrienter,


    Args:
        patch_size: the input image patch size.
        affine_shape_detector: Patch affine shape estimator, :class:`~kornia.feature.PatchAffineShapeEstimator`.
        preserve_orientation: if True, the original orientation is preserved.
    """  # pylint: disable

    def __init__(
        self, patch_size: int = 32, affine_shape_detector: Optional[nn.Module] = None, preserve_orientation: bool = True
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.affine_shape_detector = affine_shape_detector or PatchAffineShapeEstimator(self.patch_size)
        self.preserve_orientation = preserve_orientation
        if preserve_orientation:
            warnings.warn(
                "`LAFAffineShapeEstimator` default behaviour is changed "
                "and now it does preserve original LAF orientation. "
                "Make sure your code accounts for this.",
                DeprecationWarning,
                stacklevel=2,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(patch_size={self.patch_size}, "
            f"affine_shape_detector={self.affine_shape_detector}, "
            f"preserve_orientation={self.preserve_orientation})"
        )

    def forward(self, laf: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            LAF: :math:`(B, N, 2, 3)`
            img: :math:`(B, 1, H, W)`

        Returns:
            LAF_out: :math:`(B, N, 2, 3)`
        """
        KORNIA_CHECK_LAF(laf)
        KORNIA_CHECK_SHAPE(img, ["B", "1", "H", "W"])
        B, N = laf.shape[:2]
        PS: int = self.patch_size
        patches: torch.Tensor = extract_patches_from_pyramid(img, make_upright(laf), PS, True).view(-1, 1, PS, PS)
        ellipse_shape: torch.Tensor = self.affine_shape_detector(patches)
        ellipses = torch.cat([laf.view(-1, 2, 3)[..., 2].unsqueeze(1), ellipse_shape], dim=2).view(B, N, 5)
        scale_orig = get_laf_scale(laf)
        if self.preserve_orientation:
            ori_orig = get_laf_orientation(laf)
        laf_out = ellipse_to_laf(ellipses)
        ellipse_scale = get_laf_scale(laf_out)
        laf_out = scale_laf(laf_out, scale_orig / ellipse_scale)
        if self.preserve_orientation:
            laf_out = set_laf_orientation(laf_out, ori_orig)
        return laf_out

