class MultiScaleData(Data):
    def __init__(
        self,
        x=None,
        y=None,
        pos=None,
        multiscale: Optional[List[Data]] = None,
        upsample: Optional[List[Data]] = None,
        **kwargs,
    ):
        super().__init__(x=x, y=y, pos=pos, multiscale=multiscale, upsample=upsample, **kwargs)

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor and Data attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            if torch.is_tensor(item):
                self[key] = func(item)
        for scale in range(self.num_scales):
            self.multiscale[scale] = self.multiscale[scale].apply(func)

        for up in range(self.num_upsample):
            self.upsample[up] = self.upsample[up].apply(func)
        return self

    @property
    def num_scales(self):
        """ Number of scales in the multiscale array
        """
        return len(self.multiscale) if hasattr(self, "multiscale") and self.multiscale else 0

    @property
    def num_upsample(self):
        """ Number of upsample operations
        """
        return len(self.upsample) if hasattr(self, "upsample") and self.upsample else 0

    @classmethod
    def from_data(cls, data):
        ms_data = cls()
        for k, item in data:
            ms_data[k] = item
        return ms_data

