    def loss_by_feat(self, key_track_feats: Tensor, ref_track_feats: Tensor,
                     key_sampling_results: List[SamplingResult],
                     ref_sampling_results: List[SamplingResult],
                     gt_match_indices_list: List[Tensor]) -> dict:
        """Calculate the track loss and the auxiliary track loss.

        Args:
            key_track_feats (Tensor): Embeds of positive bboxes in sampling
                results of key image.
            ref_track_feats (Tensor): Embeds of all bboxes in sampling results
                of the reference image.
            key_sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            ref_sampling_results (List[obj:SamplingResults]): Assign results of
                all reference images in a batch after sampling.
            gt_match_indices_list (list(Tensor)): Mapping from instances_ids
                from key image to reference image of the same tracklet in a
                pair of images.

        Returns:
            Dict [str: Tensor]: Calculation results.
            Containing the following list of Tensors:

                - loss_track (Tensor): Results of loss_track function.
                - loss_track_aux (Tensor): Results of loss_track_aux function.
        """
        dists, cos_dists = self.match(key_track_feats, ref_track_feats,
                                      key_sampling_results,
                                      ref_sampling_results)
        targets, weights = self.get_targets(gt_match_indices_list,
                                            key_sampling_results,
                                            ref_sampling_results)
        losses = dict()

        loss_track = 0.
        loss_track_aux = 0.
        for _dists, _cos_dists, _targets, _weights in zip(
                dists, cos_dists, targets, weights):
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=_weights.sum())
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses['loss_track'] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses['loss_track_aux'] = loss_track_aux / len(dists)

        return losses