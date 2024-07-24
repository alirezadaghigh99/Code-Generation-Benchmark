class ReIDDataSample(BaseDataElement):
    """A data structure interface of ReID task.

    It's used as interfaces between different components.

    Meta field:
        img_shape (Tuple): The shape of the corresponding input image.
            Used for visualization.
        ori_shape (Tuple): The original shape of the corresponding image.
            Used for visualization.
        num_classes (int): The number of all categories.
            Used for label format conversion.

    Data field:
        gt_label (LabelData): The ground truth label.
        pred_label (LabelData): The predicted label.
        scores (torch.Tensor): The outputs of model.
    """

    @property
    def gt_label(self):
        return self._gt_label

    @gt_label.setter
    def gt_label(self, value: LabelData):
        self.set_field(value, '_gt_label', dtype=LabelData)

    @gt_label.deleter
    def gt_label(self):
        del self._gt_label

    def set_gt_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> 'ReIDDataSample':
        """Set label of ``gt_label``."""
        label = format_label(value, self.get('num_classes'))
        if 'gt_label' in self:  # setting for the second time
            self.gt_label.label = label.label
        else:  # setting for the first time
            self.gt_label = label
        return self

    def set_gt_score(self, value: torch.Tensor) -> 'ReIDDataSample':
        """Set score of ``gt_label``."""
        assert isinstance(value, torch.Tensor), \
            f'The value should be a torch.Tensor but got {type(value)}.'
        assert value.ndim == 1, \
            f'The dims of value should be 1, but got {value.ndim}.'

        if 'num_classes' in self:
            assert value.size(0) == self.num_classes, \
                f"The length of value ({value.size(0)}) doesn't "\
                f'match the num_classes ({self.num_classes}).'
            metainfo = {'num_classes': self.num_classes}
        else:
            metainfo = {'num_classes': value.size(0)}

        if 'gt_label' in self:  # setting for the second time
            self.gt_label.score = value
        else:  # setting for the first time
            self.gt_label = LabelData(score=value, metainfo=metainfo)
        return self

    @property
    def pred_feature(self):
        return self._pred_feature

    @pred_feature.setter
    def pred_feature(self, value: torch.Tensor):
        self.set_field(value, '_pred_feature', dtype=torch.Tensor)

    @pred_feature.deleter
    def pred_feature(self):
        del self._pred_feature

