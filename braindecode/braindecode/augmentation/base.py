class AugmentedDataLoader(DataLoader):
    """A base dataloader class customized to applying augmentation Transforms.

    Parameters
    ----------
    dataset : BaseDataset
        The dataset containing the signals.
    transforms : list | Transform, optional
        Transform or sequence of Transform to be applied to each batch.
    device : str | torch.device | None, optional
        Device on which to transform the data. Defaults to None.
    **kwargs : dict, optional
        keyword arguments to pass to standard DataLoader class.
    """

    def __init__(self, dataset, transforms=None, device=None, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "collate_fn cannot be used in this context because it is used "
                "to pass transform"
            )
        if transforms is None or (
            isinstance(transforms, list) and len(transforms) == 0
        ):
            self.collated_tr = _make_collateable(IdentityTransform(), device=device)
        elif isinstance(transforms, (Transform, nn.Module)):
            self.collated_tr = _make_collateable(transforms, device=device)
        elif isinstance(transforms, list):
            self.collated_tr = _make_collateable(Compose(transforms), device=device)
        else:
            raise TypeError(
                "transforms can be either a Transform object "
                "or a list of Transform objects."
            )

        super().__init__(dataset, collate_fn=self.collated_tr, **kwargs)

class Compose(Transform):
    """Transform composition.

    Callable class allowing to cast a sequence of Transform objects into a
    single one.

    Parameters
    ----------
    transforms: list
        Sequence of Transforms to be composed.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        super().__init__()

    def forward(self, X, y):
        for transform in self.transforms:
            X, y = transform(X, y)
        return X, y

