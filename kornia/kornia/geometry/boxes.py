def from_tensor(
        cls, boxes: torch.Tensor | list[torch.Tensor], mode: str = "xyxy", validate_boxes: bool = True
    ) -> Boxes:
        r"""Helper method to easily create :class:`Boxes` from boxes stored in another format.

        Args:
            boxes: 2D boxes, shape of :math:`(N, 4)`, :math:`(B, N, 4)`, :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.
            mode: The format in which the boxes are provided.

                * 'xyxy': boxes are assumed to be in the format ``xmin, ymin, xmax, ymax`` where ``width = xmax - xmin``
                  and ``height = ymax - ymin``. With shape :math:`(N, 4)`, :math:`(B, N, 4)`.
                * 'xyxy_plus': similar to 'xyxy' mode but where box width and length are defined as
                  ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``.
                  With shape :math:`(N, 4)`, :math:`(B, N, 4)`.
                * 'xywh': boxes are assumed to be in the format ``xmin, ymin, width, height`` where
                  ``width = xmax - xmin`` and ``height = ymax - ymin``. With shape :math:`(N, 4)`, :math:`(B, N, 4)`.
                * 'vertices': boxes are defined by their vertices points in the following ``clockwise`` order:
                  *top-left, top-right, bottom-right, bottom-left*. Vertices coordinates are in (x,y) order. Finally,
                  box width and height are defined as ``width = xmax - xmin`` and ``height = ymax - ymin``.
                  With shape :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.
                * 'vertices_plus': similar to 'vertices' mode but where box width and length are defined as
                  ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``. ymin + 1``.
                  With shape :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.

            validate_boxes: check if boxes are valid rectangles or not. Valid rectangles are those with width
                and height >= 1 (>= 2 when mode ends with '_plus' suffix).

        Returns:
            :class:`Boxes` class containing the original `boxes` in the format specified by ``mode``.

        Examples:
            >>> boxes_xyxy = torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]])
            >>> boxes = Boxes.from_tensor(boxes_xyxy, mode='xyxy')
            >>> boxes.data  # (2, 4, 2)
            tensor([[[0., 3.],
                     [0., 3.],
                     [0., 3.],
                     [0., 3.]],
            <BLANKLINE>
                    [[5., 1.],
                     [7., 1.],
                     [7., 3.],
                     [5., 3.]]])
        """
        quadrilaterals: torch.Tensor | list[torch.Tensor]
        if isinstance(boxes, torch.Tensor):
            quadrilaterals = _boxes_to_quadrilaterals(boxes, mode=mode, validate_boxes=validate_boxes)
        else:
            quadrilaterals = [_boxes_to_quadrilaterals(box, mode, validate_boxes) for box in boxes]

        return cls(quadrilaterals, False, mode)

