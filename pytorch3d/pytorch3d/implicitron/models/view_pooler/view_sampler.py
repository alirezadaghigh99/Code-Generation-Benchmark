class ViewSampler(Configurable, torch.nn.Module):
    """
    Implements sampling of image-based features at the 2d projections of a set
    of 3D points.

    Args:
        masked_sampling: If `True`, the `sampled_masks` output of `self.forward`
            contains the input `masks` sampled at the 2d projections. Otherwise,
            all entries of `sampled_masks` are set to 1.
        sampling_mode: Controls the mode of the `torch.nn.functional.grid_sample`
            function used to interpolate the sampled feature tensors at the
            locations of the 2d projections.
    """

    masked_sampling: bool = False
    sampling_mode: str = "bilinear"

    def forward(
        self,
        *,  # force kw args
        pts: torch.Tensor,
        seq_id_pts: Union[List[int], List[str], torch.LongTensor],
        camera: CamerasBase,
        seq_id_camera: Union[List[int], List[str], torch.LongTensor],
        feats: Dict[str, torch.Tensor],
        masks: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Project each point cloud from a batch of point clouds to corresponding
        input cameras and sample features at the 2D projection locations.

        Args:
            pts: A tensor of shape `[pts_batch x n_pts x 3]` in world coords.
            seq_id_pts: LongTensor of shape `[pts_batch]` denoting the ids of the scenes
                from which `pts` were extracted, or a list of string names.
            camera: 'n_cameras' cameras, each coresponding to a batch element of `feats`.
            seq_id_camera: LongTensor of shape `[n_cameras]` denoting the ids of the scenes
                corresponding to cameras in `camera`, or a list of string names.
            feats: a dict of tensors of per-image features `{feat_i: T_i}`.
                Each tensor `T_i` is of shape `[n_cameras x dim_i x H_i x W_i]`.
            masks: `[n_cameras x 1 x H x W]`, define valid image regions
                for sampling `feats`.
        Returns:
            sampled_feats: Dict of sampled features `{feat_i: sampled_T_i}`.
                Each `sampled_T_i` of shape `[pts_batch, n_cameras, n_pts, dim_i]`.
            sampled_masks: A tensor with  mask of the sampled features
                of shape `(pts_batch, n_cameras, n_pts, 1)`.
        """

        # convert sequence ids to long tensors
        seq_id_pts, seq_id_camera = [
            handle_seq_id(seq_id, pts.device) for seq_id in [seq_id_pts, seq_id_camera]
        ]

        if self.masked_sampling and masks is None:
            raise ValueError(
                "Masks have to be provided for `self.masked_sampling==True`"
            )

        # project pts to all cameras and sample feats from the locations of
        # the 2D projections
        sampled_feats_all_cams, sampled_masks_all_cams = project_points_and_sample(
            pts,
            feats,
            camera,
            masks if self.masked_sampling else None,
            sampling_mode=self.sampling_mode,
        )

        # generate the mask that invalidates features sampled from
        # non-corresponding cameras
        camera_pts_mask = (seq_id_camera[None] == seq_id_pts[:, None])[
            ..., None, None
        ].to(pts)

        # mask the sampled features and masks
        sampled_feats = {
            k: f * camera_pts_mask for k, f in sampled_feats_all_cams.items()
        }
        sampled_masks = sampled_masks_all_cams * camera_pts_mask

        return sampled_feats, sampled_masks

