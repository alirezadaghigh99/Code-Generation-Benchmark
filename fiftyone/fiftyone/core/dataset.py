def load_dataset(name):
    """Loads the FiftyOne dataset with the given name.

    To create a new dataset, use the :class:`Dataset` constructor.

    .. note::

        :class:`Dataset` instances are singletons keyed by their name, so all
        calls to this method with a given dataset ``name`` in a program will
        return the same object.

    Args:
        name: the name of the dataset

    Returns:
        a :class:`Dataset`
    """
    return Dataset(name, _create=False)

def from_dir(
        cls,
        dataset_dir=None,
        dataset_type=None,
        data_path=None,
        labels_path=None,
        name=None,
        persistent=False,
        overwrite=False,
        label_field=None,
        tags=None,
        dynamic=False,
        progress=None,
        **kwargs,
    ):
        """Creates a :class:`Dataset` from the contents of the given directory.

        You can create datasets with this method via the following basic
        patterns:

        (a) Provide ``dataset_dir`` and ``dataset_type`` to import the contents
            of a directory that is organized in the default layout for the
            dataset type as documented in
            :ref:`this guide <loading-datasets-from-disk>`

        (b) Provide ``dataset_type`` along with ``data_path``, ``labels_path``,
            or other type-specific parameters to perform a customized
            import. This syntax provides the flexibility to, for example,
            perform labels-only imports or imports where the source media lies
            in a different location than the labels

        In either workflow, the remaining parameters of this method can be
        provided to further configure the import.

        See :ref:`this guide <loading-datasets-from-disk>` for example usages
        of this method and descriptions of the available dataset types.

        Args:
            dataset_dir (None): the dataset directory. This can be omitted if
                you provide arguments such as ``data_path`` and ``labels_path``
            dataset_type (None): the :class:`fiftyone.types.Dataset` type of
                the dataset
            data_path (None): an optional parameter that enables explicit
                control over the location of the media for certain dataset
                types. Can be any of the following:

                -   a folder name like ``"data"`` or ``"data/"`` specifying a
                    subfolder of ``dataset_dir`` in which the media lies
                -   an absolute directory path in which the media lies. In this
                    case, the ``dataset_dir`` has no effect on the location of
                    the data
                -   a filename like ``"data.json"`` specifying the filename of
                    a JSON manifest file in ``dataset_dir`` that maps UUIDs to
                    media filepaths. Files of this format are generated when
                    passing the ``export_media="manifest"`` option to
                    :meth:`fiftyone.core.collections.SampleCollection.export`
                -   an absolute filepath to a JSON manifest file. In this case,
                    ``dataset_dir`` has no effect on the location of the data
                -   a dict mapping filenames to absolute filepaths

                By default, it is assumed that the data can be located in the
                default location within ``dataset_dir`` for the dataset type
            labels_path (None): an optional parameter that enables explicit
                control over the location of the labels. Only applicable when
                importing certain labeled dataset formats. Can be any of the
                following:

                -   a type-specific folder name like ``"labels"`` or
                    ``"labels/"`` or a filename like ``"labels.json"`` or
                    ``"labels.xml"`` specifying the location in ``dataset_dir``
                    of the labels file(s)
                -   an absolute directory or filepath containing the labels
                    file(s). In this case, ``dataset_dir`` has no effect on the
                    location of the labels

                For labeled datasets, this parameter defaults to the location
                in ``dataset_dir`` of the labels for the default layout of the
                dataset type being imported
            name (None): a name for the dataset. By default,
                :func:`get_default_dataset_name` is used
            persistent (False): whether the dataset should persist in the
                database after the session terminates
            overwrite (False): whether to overwrite an existing dataset of
                the same name
            label_field (None): controls the field(s) in which imported labels
                are stored. Only applicable if ``dataset_importer`` is a
                :class:`fiftyone.utils.data.importers.LabeledImageDatasetImporter` or
                :class:`fiftyone.utils.data.importers.LabeledVideoDatasetImporter`.
                If the importer produces a single
                :class:`fiftyone.core.labels.Label` instance per sample/frame,
                this argument specifies the name of the field to use; the
                default is ``"ground_truth"``. If the importer produces a
                dictionary of labels per sample, this argument can be either a
                string prefix to prepend to each label key or a dict mapping
                label keys to field names; the default in this case is to
                directly use the keys of the imported label dictionaries as
                field names
            tags (None): an optional tag or iterable of tags to attach to each
                sample
            dynamic (False): whether to declare dynamic attributes of embedded
                document fields that are encountered
            progress (None): whether to render a progress bar (True/False), use
                the default value ``fiftyone.config.show_progress_bars``
                (None), or a progress callback function to invoke instead
            **kwargs: optional keyword arguments to pass to the constructor of
                the :class:`fiftyone.utils.data.importers.DatasetImporter` for
                the specified ``dataset_type``

        Returns:
            a :class:`Dataset`
        """
        dataset = cls(name, persistent=persistent, overwrite=overwrite)
        dataset.add_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            data_path=data_path,
            labels_path=labels_path,
            label_field=label_field,
            tags=tags,
            dynamic=dynamic,
            progress=progress,
            **kwargs,
        )
        return dataset

