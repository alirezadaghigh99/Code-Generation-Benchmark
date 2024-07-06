def eval_batch(
    frame_data: FrameData,
    implicitron_render: ImplicitronRender,
    bg_color: Union[torch.Tensor, Sequence, str, float] = "black",
    mask_thr: float = 0.5,
    lpips_model=None,
    visualize: bool = False,
    visualize_visdom_env: str = "eval_debug",
    break_after_visualising: bool = True,
) -> Dict[str, Any]:
    """
    Produce performance metrics for a single batch of new-view synthesis
    predictions.

    Given a set of known views (for which frame_data.frame_type.endswith('known')
    is True), a new-view synthesis method (NVS) is tasked to generate new views
    of the scene from the viewpoint of the target views (for which
    frame_data.frame_type.endswith('known') is False). The resulting
    synthesized new views, stored in `implicitron_render`, are compared to the
    target ground truth in `frame_data` in terms of geometry and appearance
    resulting in a dictionary of metrics returned by the `eval_batch` function.

    Args:
        frame_data: A FrameData object containing the input to the new view
            synthesis method.
        implicitron_render: The data describing the synthesized new views.
        bg_color: The background color of the generated new views and the
            ground truth.
        lpips_model: A pre-trained model for evaluating the LPIPS metric.
        visualize: If True, visualizes the results to Visdom.

    Returns:
        results: A dictionary holding evaluation metrics.

    Throws:
        ValueError if frame_data does not have frame_type, camera, or image_rgb
        ValueError if the batch has a mix of training and test samples
        ValueError if the batch frames are not [unseen, known, known, ...]
        ValueError if one of the required fields in implicitron_render is missing
    """
    frame_type = frame_data.frame_type
    if frame_type is None:
        raise ValueError("Frame type has not been set.")

    # we check that all those fields are not None but Pyre can't infer that properly
    # TODO: assign to local variables and simplify the code.
    if frame_data.image_rgb is None:
        raise ValueError("Image is not in the evaluation batch.")

    if frame_data.camera is None:
        raise ValueError("Camera is not in the evaluation batch.")

    # eval all results in the resolution of the frame_data image
    image_resol = tuple(frame_data.image_rgb.shape[2:])

    # Post-process the render:
    # 1) check implicitron_render for Nones,
    # 2) obtain copies to make sure we dont edit the original data,
    # 3) take only the 1st (target) image
    # 4) resize to match ground-truth resolution
    cloned_render: Dict[str, torch.Tensor] = {}
    for k in ["mask_render", "image_render", "depth_render"]:
        field = getattr(implicitron_render, k)
        if field is None:
            raise ValueError(f"A required predicted field {k} is missing")

        imode = "bilinear" if k == "image_render" else "nearest"
        cloned_render[k] = (
            F.interpolate(field[:1], size=image_resol, mode=imode).detach().clone()
        )

    frame_data = copy.deepcopy(frame_data)

    # mask the ground truth depth in case frame_data contains the depth mask
    if frame_data.depth_map is not None and frame_data.depth_mask is not None:
        frame_data.depth_map *= frame_data.depth_mask

    if not isinstance(frame_type, list):  # not batch FrameData
        frame_type = [frame_type]

    is_train = is_train_frame(frame_type)
    if len(is_train) > 1 and (is_train[1] != is_train[1:]).any():
        raise ValueError(
            "All (conditioning) frames in the eval batch have to be either train/test."
        )

    for k in [
        "depth_map",
        "image_rgb",
        "fg_probability",
        "mask_crop",
    ]:
        if not hasattr(frame_data, k) or getattr(frame_data, k) is None:
            continue
        setattr(frame_data, k, getattr(frame_data, k)[:1])

    if frame_data.depth_map is None or frame_data.depth_map.sum() <= 0:
        warnings.warn("Empty or missing depth map in evaluation!")

    if frame_data.mask_crop is None:
        warnings.warn("mask_crop is None, assuming the whole image is valid.")

    if frame_data.fg_probability is None:
        warnings.warn("fg_probability is None, assuming the whole image is fg.")

    # threshold the masks to make ground truth binary masks
    mask_fg = (
        frame_data.fg_probability >= mask_thr
        if frame_data.fg_probability is not None
        # pyre-ignore [16]
        else torch.ones_like(frame_data.image_rgb[:, :1, ...]).bool()
    )

    mask_crop = (
        frame_data.mask_crop
        if frame_data.mask_crop is not None
        else torch.ones_like(mask_fg)
    )

    # unmasked g.t. image
    image_rgb = frame_data.image_rgb

    # fg-masked g.t. image
    image_rgb_masked = mask_background(
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got
        #  `Optional[torch.Tensor]`.
        frame_data.image_rgb,
        mask_fg,
        bg_color=bg_color,
    )

    # clamp predicted images
    image_render = cloned_render["image_render"].clamp(0.0, 1.0)

    if visualize:
        visualizer = _Visualizer(
            image_render=image_render,
            image_rgb_masked=image_rgb_masked,
            depth_render=cloned_render["depth_render"],
            depth_map=frame_data.depth_map,
            depth_mask=(
                frame_data.depth_mask[:1] if frame_data.depth_mask is not None else None
            ),
            visdom_env=visualize_visdom_env,
        )

    results: Dict[str, Any] = {}

    results["iou"] = iou(
        cloned_render["mask_render"],
        mask_fg,
        mask=mask_crop,
    )

    for loss_fg_mask, name_postfix in zip((mask_crop, mask_fg), ("_masked", "_fg")):

        loss_mask_now = mask_crop * loss_fg_mask

        for rgb_metric_name, rgb_metric_fun in zip(
            ("psnr", "rgb_l1"), (calc_psnr, rgb_l1)
        ):
            metric_name = rgb_metric_name + name_postfix
            results[metric_name] = rgb_metric_fun(
                image_render,
                image_rgb_masked,
                mask=loss_mask_now,
            )

            if visualize:
                visualizer.show_rgb(
                    results[metric_name].item(), metric_name, loss_mask_now
                )

        if name_postfix == "_fg" and frame_data.depth_map is not None:
            # only record depth metrics for the foreground
            _, abs_ = eval_depth(
                cloned_render["depth_render"],
                # pyre-fixme[6]: For 2nd param expected `Tensor` but got
                #  `Optional[Tensor]`.
                frame_data.depth_map,
                get_best_scale=True,
                mask=loss_mask_now,
                crop=5,
            )
            results["depth_abs" + name_postfix] = abs_.mean()

            if visualize:
                visualizer.show_depth(abs_.mean().item(), name_postfix, loss_mask_now)
                if break_after_visualising:
                    breakpoint()  # noqa: B601

    # add the rgb metrics between the render and the unmasked image
    for rgb_metric_name, rgb_metric_fun in zip(
        ("psnr_full_image", "rgb_l1_full_image"), (calc_psnr, rgb_l1)
    ):
        results[rgb_metric_name] = rgb_metric_fun(
            image_render,
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Optional[Tensor]`.
            image_rgb,
            mask=mask_crop,
        )

    if lpips_model is not None:
        for gt_image_type in ("_full_image", "_masked"):
            im1, im2 = [
                2.0 * im.clamp(0.0, 1.0) - 1.0  # pyre-ignore[16]
                for im in (
                    image_rgb_masked if gt_image_type == "_masked" else image_rgb,
                    cloned_render["image_render"],
                )
            ]
            results["lpips" + gt_image_type] = lpips_model.forward(im1, im2).item()

    # convert all metrics to floats
    results = {k: float(v) for k, v in results.items()}

    results["meta"] = {
        # store the size of the batch (corresponds to n_src_views+1)
        "batch_size": len(frame_type),
        # store the type of the target frame
        # pyre-fixme[16]: `None` has no attribute `__getitem__`.
        "frame_type": str(frame_data.frame_type[0]),
    }

    return results

