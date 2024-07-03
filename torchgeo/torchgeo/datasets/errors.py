class DatasetNotFoundError(FileNotFoundError):
    """Raised when a dataset is requested but doesn't exist.

    .. versionadded:: 0.6
    """

    def __init__(self, dataset: Dataset[object]) -> None:
        """Initialize a new DatasetNotFoundError instance.

        Args:
            dataset: The dataset that was requested.
        """
        msg = 'Dataset not found'

        if hasattr(dataset, 'root'):
            var = 'root'
            val = dataset.root
        elif hasattr(dataset, 'paths'):
            var = 'paths'
            val = dataset.paths
        else:
            super().__init__(f'{msg}.')
            return

        msg += f' in `{var}={val!r}` and '

        if hasattr(dataset, 'download') and not dataset.download:
            msg += '`download=False`'
        else:
            msg += 'cannot be automatically downloaded'

        msg += f', either specify a different `{var}` or '

        if hasattr(dataset, 'download') and not dataset.download:
            msg += 'use `download=True` to automatically'
        else:
            msg += 'manually'

        msg += ' download the dataset.'

        super().__init__(msg)