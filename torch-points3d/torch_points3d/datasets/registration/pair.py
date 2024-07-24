class MultiScalePair(Pair):
    def __init__(
            self,
            x=None,
            y=None,
            pos=None,
            multiscale: Optional[List[Data]] = None,
            upsample: Optional[List[Data]] = None,
            x_target=None,
            pos_target=None,
            multiscale_target: Optional[List[Data]] = None,
            upsample_target: Optional[List[Data]] = None,
            **kwargs,
    ):
        super(MultiScalePair, self).__init__(x=x, pos=pos,
                                             multiscale=multiscale,
                                             upsample=upsample,
                                             x_target=x_target, pos_target=pos_target,
                                             multiscale_target=multiscale_target,
                                             upsample_target=upsample_target,
                                             **kwargs)
        self.__data_class__ = MultiScaleData

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
            self.multiscale_target[scale] = self.multiscale_target[scale].apply(func)

        for up in range(self.num_upsample):
            self.upsample[up] = self.upsample[up].apply(func)
            self.upsample_target[up] = self.upsample_target[up].apply(func)
        return self

    @property
    def num_scales(self):
        """ Number of scales in the multiscale array
        """
        return len(self.multiscale) if self.multiscale else 0

    @property
    def num_upsample(self):
        """ Number of upsample operations
        """
        return len(self.upsample) if self.upsample else 0

    @classmethod
    def from_data(cls, data):
        ms_data = cls()
        for k, item in data:
            ms_data[k] = item
        return ms_data

class Pair(Data):

    def __init__(
            self,
            x=None,
            y=None,
            pos=None,
            x_target=None,
            pos_target=None,
            **kwargs,
    ):
        self.__data_class__ = Data
        super(Pair, self).__init__(x=x, pos=pos,
                                   x_target=x_target, pos_target=pos_target, **kwargs)


    @classmethod
    def make_pair(cls, data_source, data_target):
        """
        add in a Data object the source elem, the target elem.
        """
        # add concatenation of the point cloud
        batch = cls()
        for key in data_source.keys:
            batch[key] = data_source[key]
        for key_target in data_target.keys:
            batch[key_target+"_target"] = data_target[key_target]
        if(batch.x is None):
            batch["x_target"] = None
        return batch.contiguous()

    def to_data(self):
        data_source = self.__data_class__()
        data_target = self.__data_class__()
        for key in self.keys:
            match = re.search(r"(.+)_target$", key)
            if match is None:
                data_source[key] = self[key]
            else:
                new_key = match.groups()[0]
                data_target[new_key] = self[key]
        return data_source, data_target

    @property
    def num_nodes_target(self):
        for key, item in self('x_target', 'pos_target', 'norm_target', 'batch_target'):
            return item.size(self.__cat_dim__(key, item))
        return None

