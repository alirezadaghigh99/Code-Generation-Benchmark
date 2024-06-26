    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  border_value=0,
                  interpolation='bilinear'):
        """Translate the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            border_value (int | float): Border value. Default 0.
            interpolation (str): Same as :func:`mmcv.imtranslate`.

        Returns:
            Translated masks.
        """