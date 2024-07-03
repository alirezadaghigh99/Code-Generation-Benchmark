def preprocess_input(
    image_rgb: Optional[torch.Tensor],
    fg_probability: Optional[torch.Tensor],
    depth_map: Optional[torch.Tensor],
    mask_images: bool,
    mask_depths: bool,
    mask_threshold: float,
    bg_color: Tuple[float, float, float],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Helper function to preprocess the input images and optional depth maps
    to apply masking if required.

    Args:
        image_rgb: A tensor of shape `(B, 3, H, W)` containing a batch of rgb images
            corresponding to the source viewpoints from which features will be extracted
        fg_probability: A tensor of shape `(B, 1, H, W)` containing a batch
            of foreground masks with values in [0, 1].
        depth_map: A tensor of shape `(B, 1, H, W)` containing a batch of depth maps.
        mask_images: Whether or not to mask the RGB image background given the
            foreground mask (the `fg_probability` argument of `GenericModel.forward`)
        mask_depths: Whether or not to mask the depth image background given the
            foreground mask (the `fg_probability` argument of `GenericModel.forward`)
        mask_threshold: If greater than 0.0, the foreground mask is
            thresholded by this value before being applied to the RGB/Depth images
        bg_color: RGB values for setting the background color of input image
            if mask_images=True. Defaults to (0.0, 0.0, 0.0). Each renderer has its own
            way to determine the background color of its output, unrelated to this.

    Returns:
        Modified image_rgb, fg_mask, depth_map
    """
    if image_rgb is not None and image_rgb.ndim == 3:
        # The FrameData object is used for both frames and batches of frames,
        # and a user might get this error if those were confused.
        # Perhaps a user has a FrameData `fd` representing a single frame and
        # wrote something like `model(**fd)` instead of
        # `model(**fd.collate([fd]))`.
        raise ValueError(
            "Model received unbatched inputs. "
            + "Perhaps they came from a FrameData which had not been collated."
        )

    fg_mask = fg_probability
    if fg_mask is not None and mask_threshold > 0.0:
        # threshold masks
        warnings.warn("Thresholding masks!")
        fg_mask = (fg_mask >= mask_threshold).type_as(fg_mask)

    if mask_images and fg_mask is not None and image_rgb is not None:
        # mask the image
        warnings.warn("Masking images!")
        image_rgb = image_utils.mask_background(
            image_rgb, fg_mask, dim_color=1, bg_color=torch.tensor(bg_color)
        )

    if mask_depths and fg_mask is not None and depth_map is not None:
        # mask the depths
        assert (
            mask_threshold > 0.0
        ), "Depths should be masked only with thresholded masks"
        warnings.warn("Masking depths!")
        depth_map = depth_map * fg_mask

    return image_rgb, fg_mask, depth_mapdef weighted_sum_losses(
    preds: Dict[str, torch.Tensor], loss_weights: Dict[str, float]
) -> Optional[torch.Tensor]:
    """
    A helper function to compute the overall loss as the dot product
    of individual loss functions with the corresponding weights.
    """
    losses_weighted = [
        preds[k] * float(w)
        for k, w in loss_weights.items()
        if (k in preds and w != 0.0)
    ]
    if len(losses_weighted) == 0:
        warnings.warn("No main objective found.")
        return None
    loss = sum(losses_weighted)
    assert torch.is_tensor(loss)
    return loss