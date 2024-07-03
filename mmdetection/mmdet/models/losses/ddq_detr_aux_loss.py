    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
             **kwargs):
        """Calculate auxiliary branches loss for dense queries.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores, has shape (bs, num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates,
                has shape (bs, num_dense_queries, 4) with the last
                dimension arranged as (x1, y1, x2, y2).
            gt_bboxes (list[Tensor]): List of unnormalized ground truth
                bboxes for each image, each has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            gt_labels (list[Tensor]): List of ground truth classification
                index for each image, each has shape (num_gt,).
                NOTE: num_gt is dynamic for each image.
            img_metas (list[dict]): Meta information for one image,
                e.g., image size, scaling factor, etc.

        Returns:
            dict: A dictionary of loss components.
        """
        flatten_cls_scores = cls_scores
        flatten_bbox_preds = bbox_preds

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            gt_bboxes,
            img_metas,
            gt_labels_list=gt_labels,
        )
        (labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox, \
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_single,
                flatten_cls_scores,
                flatten_bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(aux_loss_cls=losses_cls, aux_loss_bbox=losses_bbox)