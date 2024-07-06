def from_torch_dataset(cls, dataset, transform=None, index_to_filename=None):
        """Builds a LightlyDataset from a PyTorch (or torchvision) dataset.

        Args:
            dataset:
                PyTorch/torchvision dataset.
            transform:
                Image transforms (as in torchvision).
            index_to_filename:
                Function which takes the dataset and index as input and returns
                the filename of the file at the index. If None, uses default.

        Returns:
            A LightlyDataset object.

        Examples:
            >>> # load cifar10 from torchvision
            >>> import torchvision
            >>> import lightly.data as data
            >>> base = torchvision.datasets.CIFAR10(root='./')
            >>> dataset = data.LightlyDataset.from_torch_dataset(base)

        """
        # create an "empty" dataset object
        dataset_obj = cls(
            None,
            index_to_filename=index_to_filename,
        )

        # populate it with the torch dataset
        if transform is not None:
            dataset.transform = transform
            # If dataset is a VisionDataset, we need to initialize transforms, too.
            if isinstance(dataset, VisionDataset):
                dataset.transforms = StandardTransform(
                    transform, dataset.target_transform
                )
        dataset_obj.dataset = dataset
        return dataset_obj

