def label_to_onehot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert the label-format input to one-hot.

        Args:
            label (torch.Tensor): The label-format input. The format
                of item must be label-format.
            num_classes (int): The number of classes.

        Returns:
            torch.Tensor: The converted results.
        """
        assert isinstance(label, torch.Tensor)
        onehot = label.new_zeros((num_classes, ))
        assert max(label, default=torch.tensor(0)).item() < num_classes
        onehot[label] = 1
        return onehot

