class SetCriterion(nn.Module):
    """
    This class computes the loss for YOLOv5.

    Args:
        num_anchors (int): The number of anchors.
        num_classes (int): The number of output classes of the model.
        fl_gamma (float): focal loss gamma (efficientDet default gamma=1.5). Default: 0.0.
        box_gain (float): box loss gain. Default: 0.05.
        cls_gain (float): class loss gain. Default: 0.5.
        cls_pos (float): cls BCELoss positive_weight. Default: 1.0.
        obj_gain (float): obj loss gain (scale with pixels). Default: 1.0.
        obj_pos (float): obj BCELoss positive_weight. Default: 1.0.
        anchor_thresh (float): anchor-multiple threshold. Default: 4.0.
        label_smoothing (float): Label smoothing epsilon. Default: 0.0.
        auto_balance (bool): Auto balance. Default: False.
    """

    def __init__(
        self,
        strides: List[int],
        anchor_grids: List[List[float]],
        num_classes: int,
        fl_gamma: float = 0.0,
        box_gain: float = 0.05,
        cls_gain: float = 0.5,
        cls_pos: float = 1.0,
        obj_gain: float = 1.0,
        obj_pos: float = 1.0,
        anchor_thresh: float = 4.0,
        label_smoothing: float = 0.0,
        auto_balance: bool = False,
    ) -> None:
        super().__init__()
        assert len(strides) == len(anchor_grids)

        self.num_classes = num_classes
        self.strides = strides
        self.anchor_grids = anchor_grids
        self.num_anchors = len(anchor_grids[0]) // 2

        self.balance = [4.0, 1.0, 0.4]
        self.ssi = 0  # stride 16 index

        self.sort_obj_iou = False

        # Define criteria
        self.cls_pos = cls_pos
        self.obj_pos = obj_pos

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # positive, negative BCE targets
        smooth_bce = det_utils.smooth_binary_cross_entropy(eps=label_smoothing)
        self.smooth_pos = smooth_bce[0]
        self.smooth_neg = smooth_bce[1]

        # Parameters for training
        self.gr = 1.0
        self.auto_balance = auto_balance
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.obj_gain = obj_gain
        self.anchor_thresh = anchor_thresh

    def forward(
        self,
        targets: Tensor,
        head_outputs: List[Tensor],
    ) -> Dict[str, Tensor]:
        """
        This performs the loss computation.

        Args:
            targets (Tensor): list of dicts, such that len(targets) == batch_size. The
                expected keys in each dict depends on the losses applied, see each loss' doc
            head_outputs (List[Tensor]): dict of tensors, see the output specification
                of the model for the format
        """
        device = targets.device
        anchor_grids = torch.as_tensor(self.anchor_grids, dtype=torch.float32, device=device).view(
            self.num_anchors, -1, 2
        )
        strides = torch.as_tensor(self.strides, dtype=torch.float32, device=device).view(-1, 1, 1)
        anchor_grids /= strides

        target_cls, target_box, indices, anchors = self.build_targets(targets, head_outputs, anchor_grids)

        pos_weight_cls = torch.as_tensor([self.cls_pos], device=device)
        pos_weight_obj = torch.as_tensor([self.obj_pos], device=device)

        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        # Computing the losses
        for i, pred_logits in enumerate(head_outputs):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            target_obj = torch.zeros_like(pred_logits[..., 0], device=device)  # target obj

            num_targets = b.shape[0]  # number of targets
            if num_targets > 0:
                # prediction subset corresponding to targets
                pred_logits_subset = pred_logits[b, a, gj, gi]

                # Regression
                pred_box = det_utils.encode_single(pred_logits_subset, anchors[i])
                iou = det_utils.bbox_iou(pred_box.T, target_box[i], x1y1x2y2=False)
                loss_box += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).to(dtype=target_obj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id]
                    score_iou = score_iou[sort_id]
                target_obj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    # targets
                    t = torch.full_like(pred_logits_subset[:, 5:], self.smooth_neg, device=device)
                    t[torch.arange(num_targets), target_cls[i]] = self.smooth_pos
                    loss_cls += F.binary_cross_entropy_with_logits(
                        pred_logits_subset[:, 5:], t, pos_weight=pos_weight_cls
                    )

            obji = F.binary_cross_entropy_with_logits(
                pred_logits[..., 4], target_obj, pos_weight=pos_weight_obj
            )
            loss_obj += obji * self.balance[i]  # obj loss
            if self.auto_balance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.auto_balance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        loss_box *= self.box_gain
        loss_obj *= self.obj_gain
        loss_cls *= self.cls_gain

        return {
            "cls_logits": loss_cls,
            "bbox_regression": loss_box,
            "objectness": loss_obj,
        }

    def build_targets(
        self,
        targets: Tensor,
        head_outputs: List[Tensor],
        anchor_grids: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tuple[Tensor, Tensor, Tensor, Tensor]], List[Tensor]]:
        device = targets.device
        num_anchors = self.num_anchors

        num_targets = targets.shape[0]

        gain = torch.ones(7, device=device).long()  # normalized to gridspace gain
        # same as .repeat_interleave(num_targets)
        ai = torch.arange(num_anchors, device=device).float().view(num_anchors, 1).repeat(1, num_targets)
        # append anchor indices
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), ai[:, :, None]), 2)

        g_bias = 0.5
        offset = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=device,
            ).float()
            * g_bias
        )  # offsets

        target_cls, target_box, anch = [], [], []
        indices: List[Tuple[Tensor, Tensor, Tensor, Tensor]] = []

        for i in range(num_anchors):
            anchors = anchor_grids[i]
            gain[2:6] = torch.tensor(head_outputs[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            targets_with_gain = targets * gain
            if num_targets > 0:
                # Matches
                r = targets_with_gain[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_thresh  # compare
                # j = wh_iou(anchors, targets_with_gain[:, 4:6]) > model.hyp['iou_t']
                # iou(3, n) = wh_iou(anchors(3, 2), gwh(n, 2))
                targets_with_gain = targets_with_gain[j]  # filter

                # Offsets
                gxy = targets_with_gain[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                idx_jk = ((gxy % 1.0 < g_bias) & (gxy > 1.0)).T
                idx_lm = ((gxi % 1.0 < g_bias) & (gxi > 1.0)).T
                j = torch.stack(
                    (
                        torch.ones_like(idx_jk[0]),
                        idx_jk[0],
                        idx_jk[1],
                        idx_lm[0],
                        idx_lm[1],
                    )
                )
                targets_with_gain = targets_with_gain.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + offset[:, None])[j]
            else:
                targets_with_gain = targets[0]
                offsets = torch.tensor(0, device=device)

            # Define
            idx_bc = targets_with_gain[:, :2].long().T  # image, class
            gxy = targets_with_gain[:, 2:4]  # grid xy
            gwh = targets_with_gain[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            idx_gij = gij.T  # grid xy indices

            # Append
            a = targets_with_gain[:, 6].long()  # anchor indices
            # image, anchor, grid indices
            indices.append(
                (
                    idx_bc[0],
                    a,
                    idx_gij[1].clamp_(0, gain[3] - 1),
                    idx_gij[0].clamp_(0, gain[2] - 1),
                )
            )
            target_box.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            target_cls.append(idx_bc[1])  # class

        return target_cls, target_box, indices, anch