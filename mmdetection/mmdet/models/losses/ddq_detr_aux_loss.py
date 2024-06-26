    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    img_metas,
                    gt_labels_list=None,
                    **kwargs):
        """Compute regression and classification targets for a batch images.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores, has shape (bs, num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates,
                has shape (bs, num_dense_queries, 4) with the last
                dimension arranged as (x1, y1, x2, y2).
            gt_bboxes_list (List[Tensor]): List of unnormalized ground truth
                bboxes for each image, each has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            img_metas (list[dict]): Meta information for one image,
                e.g., image size, scaling factor, etc.
            gt_labels_list (list[Tensor]): List of ground truth classification
                    index for each image, each has shape (num_gt,).
                    NOTE: num_gt is dynamic for each image.
                    Default: None.

        Returns:
            tuple: a tuple containing the following targets.

            - all_labels (list[Tensor]): Labels for all images.
            - all_label_weights (list[Tensor]): Label weights for all images.
            - all_bbox_targets (list[Tensor]): Bbox targets for all images.
            - all_assign_metrics (list[Tensor]): Normalized alignment metrics
                for all images.
        """
        (all_labels, all_label_weights, all_bbox_targets,
         all_assign_metrics) = multi_apply(self._get_target_single, cls_scores,
                                           bbox_preds, gt_bboxes_list,
                                           gt_labels_list, img_metas)

        return (all_labels, all_label_weights, all_bbox_targets,
                all_assign_metrics)