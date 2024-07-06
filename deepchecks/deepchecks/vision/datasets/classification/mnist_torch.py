def load_dataset(train: bool = True, batch_size: t.Optional[int] = None, shuffle: bool = False, pin_memory: bool = True,
                 object_type: Literal['VisionData', 'DataLoader'] = 'DataLoader', use_iterable_dataset: bool = False,
                 n_samples=None, device: t.Union[str, torch.device] = 'cpu') -> t.Union[DataLoader, VisionData]:
    """Download MNIST dataset.

    Parameters
    ----------
    train : bool, default : True
        Train or Test dataset
    batch_size: int, optional
        how many samples per batch to load
    shuffle : bool , default : False
        to reshuffled data at every epoch or not, cannot work with use_iterable_dataset=True
    pin_memory : bool, default : True
        If ``True``, the data loader will copy Tensors
        into CUDA pinned memory before returning them.
    object_type : Literal[Dataset, DataLoader], default 'DataLoader'
        object type to return. if `'VisionData'` then :obj:`deepchecks.vision.VisionData`
        will be returned, if `'DataLoader'` then :obj:`torch.utils.data.DataLoader`
    use_iterable_dataset : bool, default False
        if True, will use :obj:`IterableTorchMnistDataset` instead of :obj:`TorchMnistDataset`
    n_samples : int, optional
        Only relevant for loading a VisionData. Number of samples to load. Return the first n_samples if shuffle
        is False otherwise selects n_samples at random. If None, returns all samples.
    device : t.Union[str, torch.device], default : 'cpu'
        device to use in tensor calculations
    Returns
    -------
    Union[:obj:`deepchecks.vision.VisionData`, :obj:`torch.utils.data.DataLoader`]

        depending on the ``object_type`` parameter value, instance of
        :obj:`deepchecks.vision.VisionData` or :obj:`torch.utils.data.DataLoader`
        will be returned

    """
    batch_size = batch_size or (64 if train else 1000)
    transform = A.Compose([A.Normalize(mean=(0.1307,), std=(0.3081,)), ToTensorV2()])
    if use_iterable_dataset:
        dataset = IterableTorchMnistDataset(train=train, transform=transform, n_samples=n_samples)
    else:
        dataset = TorchMnistDataset(str(MNIST_DIR), train=train, download=True, transform=transform)

    if object_type == 'DataLoader':
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                          generator=torch.Generator())
    elif object_type == 'VisionData':
        model = load_model(device=device)
        loader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, generator=torch.Generator(),
                            collate_fn=deepchecks_collate(model))
        if not use_iterable_dataset:
            loader = get_data_loader_sequential(loader, shuffle, n_samples)
        return VisionData(loader, task_type='classification', reshuffle_data=False)
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')

def collate_without_model(data) -> t.Tuple[t.List[np.ndarray], t.List[int]]:
    """Collate function for the mnist dataset returning images and labels in correct format as tuple."""
    raw_images = torch.stack([x[0] for x in data])
    labels = [x[1] for x in data]
    images = raw_images.permute(0, 2, 3, 1)
    images = un_normalize_batch(images, mean=(0.1307,), std=(0.3081,))
    return images, labels

