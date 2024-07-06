def load_zoo_dataset(
    name,
    split=None,
    splits=None,
    label_field=None,
    dataset_name=None,
    dataset_dir=None,
    download_if_necessary=True,
    drop_existing_dataset=False,
    persistent=False,
    overwrite=False,
    cleanup=True,
    progress=None,
    **kwargs,
):
    """Loads the dataset of the given name from the FiftyOne Dataset Zoo as
    a :class:`fiftyone.core.dataset.Dataset`.

    By default, the dataset will be downloaded if it does not already exist in
    the specified directory.

    If you do not specify a custom ``dataset_name`` and you have previously
    loaded the same zoo dataset and split(s) into FiftyOne, the existing
    :class:`fiftyone.core.dataset.Dataset` will be returned.

    Args:
        name: the name of the zoo dataset to load. Call
            :func:`list_zoo_datasets` to see the available datasets
        split (None) a split to load, if applicable. Typical values are
            ``("train", "validation", "test")``. If neither ``split`` nor
            ``splits`` are provided, all available splits are loaded. Consult
            the documentation for the :class:`ZooDataset` you specified to see
            the supported splits
        splits (None): a list of splits to load, if applicable. Typical values
            are ``("train", "validation", "test")``. If neither ``split`` nor
            ``splits`` are provided, all available splits are loaded. Consult
            the documentation for the :class:`ZooDataset` you specified to see
            the supported splits
        label_field (None): the label field (or prefix, if the dataset contains
            multiple label fields) in which to store the dataset's labels. By
            default, this is ``"ground_truth"`` if the dataset contains a
            single label field. If the dataset contains multiple label fields
            and this value is not provided, the labels will be stored under
            dataset-specific field names
        dataset_name (None): an optional name to give the returned
            :class:`fiftyone.core.dataset.Dataset`. By default, a name will be
            constructed based on the dataset and split(s) you are loading
        dataset_dir (None): the directory in which the dataset is stored or
            will be downloaded. By default, the dataset will be located in
            ``fiftyone.config.dataset_zoo_dir``
        download_if_necessary (True): whether to download the dataset if it is
            not found in the specified dataset directory
        drop_existing_dataset (False): whether to drop an existing dataset
            with the same name if it exists
        persistent (False): whether the dataset should persist in the database
            after the session terminates
        overwrite (False): whether to overwrite any existing files if the
            dataset is to be downloaded
        cleanup (True): whether to cleanup any temporary files generated during
            download
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead
        **kwargs: optional arguments to pass to the
            :class:`fiftyone.utils.data.importers.DatasetImporter` constructor.
            If ``download_if_necessary == True``, then ``kwargs`` can also
            contain arguments for :func:`download_zoo_dataset`

    Returns:
        a :class:`fiftyone.core.dataset.Dataset`
    """
    splits = _parse_splits(split, splits)

    if download_if_necessary:
        zoo_dataset_cls = _get_zoo_dataset_cls(name)
        download_kwargs, _ = fou.extract_kwargs_for_class(
            zoo_dataset_cls, kwargs
        )

        info, dataset_dir = download_zoo_dataset(
            name,
            splits=splits,
            dataset_dir=dataset_dir,
            overwrite=overwrite,
            cleanup=cleanup,
            **download_kwargs,
        )
        zoo_dataset = info.get_zoo_dataset()
    else:
        download_kwargs = {}
        zoo_dataset, dataset_dir = _parse_dataset_details(name, dataset_dir)
        info = zoo_dataset.load_info(dataset_dir, warn_deprecated=True)

    dataset_type = info.get_dataset_type()
    dataset_importer_cls = dataset_type.get_dataset_importer_cls()

    #
    # For unlabeled (e.g., test) splits, some importers need to be explicitly
    # told to generate samples for media with no corresponding labels entry.
    #
    # By convention, all such importers use `include_all_data` for this flag.
    # If a new zoo dataset is added that requires a different customized
    # parameter, we'd need to improve this logic here
    #
    kwargs["include_all_data"] = True

    importer_kwargs, unused_kwargs = fou.extract_kwargs_for_class(
        dataset_importer_cls, kwargs
    )

    # Inject default importer kwargs, if any
    if zoo_dataset.importer_kwargs:
        for key, value in zoo_dataset.importer_kwargs.items():
            if key not in importer_kwargs:
                importer_kwargs[key] = value

    for key, value in unused_kwargs.items():
        if (
            key in download_kwargs
            or key == "include_all_data"
            or value is None
        ):
            continue

        logger.warning(
            "Ignoring unsupported parameter '%s' for importer type %s",
            key,
            dataset_importer_cls,
        )

    if dataset_name is None:
        dataset_name = zoo_dataset.name
        if splits is not None:
            dataset_name += "-" + "-".join(splits)

        if "max_samples" in importer_kwargs:
            dataset_name += "-%s" % importer_kwargs["max_samples"]

    if fo.dataset_exists(dataset_name):
        if not drop_existing_dataset:
            logger.info(
                "Loading existing dataset '%s'. To reload from disk, either "
                "delete the existing dataset or provide a custom "
                "`dataset_name` to use",
                dataset_name,
            )
            return fo.load_dataset(dataset_name)

        logger.info("Deleting existing dataset '%s'", dataset_name)
        fo.delete_dataset(dataset_name)

    if splits is None and zoo_dataset.has_splits:
        splits = zoo_dataset.supported_splits

    dataset = fo.Dataset(dataset_name, persistent=persistent)

    if splits:
        for split in splits:
            if not zoo_dataset.has_split(split):
                raise ValueError(
                    "Invalid split '%s'; supported values are %s"
                    % (split, zoo_dataset.supported_splits)
                )

        for split in splits:
            logger.info("Loading '%s' split '%s'", zoo_dataset.name, split)
            split_dir = zoo_dataset.get_split_dir(dataset_dir, split)
            dataset_importer, _ = foud.build_dataset_importer(
                dataset_type, dataset_dir=split_dir, **importer_kwargs
            )
            dataset.add_importer(
                dataset_importer,
                label_field=label_field,
                tags=[split],
                progress=progress,
            )
    else:
        logger.info("Loading '%s'", zoo_dataset.name)
        dataset_importer, _ = foud.build_dataset_importer(
            dataset_type, dataset_dir=dataset_dir, **importer_kwargs
        )
        dataset.add_importer(
            dataset_importer,
            label_field=label_field,
            progress=progress,
        )

    if info.classes is not None and not dataset.default_classes:
        dataset.default_classes = info.classes

    logger.info("Dataset '%s' created", dataset.name)

    return dataset

