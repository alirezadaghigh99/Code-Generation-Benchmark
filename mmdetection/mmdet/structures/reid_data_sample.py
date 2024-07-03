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