def default_dataset_location(dataset_name: str) -> Path:
    """Returns the default download location for Avalanche datasets.

    The default value is "~/.avalanche/data/<dataset_name>", but it may be
    changed via the `dataset_location` value in the configuration file
    in `~/.avalanche/config.json`.

    :param dataset_name: The name of the dataset. Must be a string that
        can be used to name a directory in most filesystems!
    :return: The default path for the dataset.
    """
    base_dir = os.path.expanduser(AVALANCHE_CONFIG["dataset_location"])
    return Path(f"{base_dir}/{dataset_name}")