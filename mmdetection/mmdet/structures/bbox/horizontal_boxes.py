    def is_inside(self,
                  img_shape: Tuple[int, int],
                  all_inside: bool = False,
                  allowed_border: int = 0) -> BoolTensor:
        """Find boxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image shape
                boundary by more than ``allowed_border`` are considered
                "outside" Defaults to 0.
        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, 4),
            the output has shape (m, n).
        """
        img_h, img_w = img_shape
        boxes = self.tensor
        if all_inside:
            return (boxes[:, 0] >= -allowed_border) & \
                (boxes[:, 1] >= -allowed_border) & \
                (boxes[:, 2] < img_w + allowed_border) & \
                (boxes[:, 3] < img_h + allowed_border)
        else:
            return (boxes[..., 0] < img_w + allowed_border) & \
                (boxes[..., 1] < img_h + allowed_border) & \
                (boxes[..., 2] > -allowed_border) & \
                (boxes[..., 3] > -allowed_border)