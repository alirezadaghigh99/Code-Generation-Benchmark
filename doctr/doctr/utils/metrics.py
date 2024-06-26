class LocalizationConfusion:
    r"""Implements common confusion metrics and mean IoU for localization evaluation.

    The aggregated metrics are computed as follows:

    .. math::
        \forall Y \in \mathcal{B}^N, \forall X \in \mathcal{B}^M, \\
        Recall(X, Y) = \frac{1}{N} \sum\limits_{i=1}^N g_{X}(Y_i) \\
        Precision(X, Y) = \frac{1}{M} \sum\limits_{i=1}^M g_{X}(Y_i) \\
        meanIoU(X, Y) = \frac{1}{M} \sum\limits_{i=1}^M \max\limits_{j \in [1, N]}  IoU(X_i, Y_j)

    with the function :math:`IoU(x, y)` being the Intersection over Union between bounding boxes :math:`x` and
    :math:`y`, and the function :math:`g_{X}` defined as:

    .. math::
        \forall y \in \mathcal{B},
        g_X(y) = \left\{
            \begin{array}{ll}
                1 & \mbox{if } y\mbox{ has been assigned to any }(X_i)_i\mbox{ with an }IoU \geq 0.5 \\
                0 & \mbox{otherwise.}
            \end{array}
        \right.

    where :math:`\mathcal{B}` is the set of possible bounding boxes,
    :math:`N` (number of ground truths) and :math:`M` (number of predictions) are strictly positive integers.

    >>> import numpy as np
    >>> from doctr.utils import LocalizationConfusion
    >>> metric = LocalizationConfusion(iou_thresh=0.5)
    >>> metric.update(np.asarray([[0, 0, 100, 100]]), np.asarray([[0, 0, 70, 70], [110, 95, 200, 150]]))
    >>> metric.summary()

    Args:
    ----
        iou_thresh: minimum IoU to consider a pair of prediction and ground truth as a match
        use_polygons: if set to True, predictions and targets will be expected to have rotated format
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        use_polygons: bool = False,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.use_polygons = use_polygons
        self.reset()

    def update(self, gts: np.ndarray, preds: np.ndarray) -> None:
        """Updates the metric

        Args:
        ----
            gts: a set of relative bounding boxes either of shape (N, 4) or (N, 5) if they are rotated ones
            preds: a set of relative bounding boxes either of shape (M, 4) or (M, 5) if they are rotated ones
        """
        if preds.shape[0] > 0:
            # Compute IoU
            if self.use_polygons:
                iou_mat = polygon_iou(gts, preds)
            else:
                iou_mat = box_iou(gts, preds)
            self.tot_iou += float(iou_mat.max(axis=0).sum())

            # Assign pairs
            gt_indices, pred_indices = linear_sum_assignment(-iou_mat)
            self.matches += int((iou_mat[gt_indices, pred_indices] >= self.iou_thresh).sum())

        # Update counts
        self.num_gts += gts.shape[0]
        self.num_preds += preds.shape[0]

    def summary(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Computes the aggregated metrics

        Returns
        -------
            a tuple with the recall, precision and meanIoU scores
        """
        # Recall
        recall = self.matches / self.num_gts if self.num_gts > 0 else None

        # Precision
        precision = self.matches / self.num_preds if self.num_preds > 0 else None

        # mean IoU
        mean_iou = round(self.tot_iou / self.num_preds, 2) if self.num_preds > 0 else None

        return recall, precision, mean_iou

    def reset(self) -> None:
        self.num_gts = 0
        self.num_preds = 0
        self.matches = 0
        self.tot_iou = 0.0