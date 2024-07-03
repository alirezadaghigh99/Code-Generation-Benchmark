class Sample(_SampleMixin, Document, metaclass=SampleSingleton):
    """A sample in a :class:`fiftyone.core.dataset.Dataset`.

    Samples store all information associated with a particular piece of data in
    a dataset, including basic metadata about the data, one or more sets of
    labels (ground truth, user-provided, or FiftyOne-generated), and additional
    features associated with subsets of the data and/or label sets.

    .. note::

        :class:`Sample` instances that are **in datasets** are singletons,
        i.e.,  ``dataset[sample_id]`` will always return the same
        :class:`Sample` instance.

    Args:
        filepath: the path to the data on disk. The path is converted to an
            absolute path (if necessary) via
            :func:`fiftyone.core.storage.normalize_path`
        tags (None): a list of tags for the sample
        metadata (None): a :class:`fiftyone.core.metadata.Metadata` instance
        **kwargs: additional fields to dynamically set on the sample
    """

    _NO_DATASET_DOC_CLS = foo.NoDatasetSampleDocument

    def __init__(self, filepath, tags=None, metadata=None, **kwargs):
        super().__init__(
            filepath=filepath, tags=tags, metadata=metadata, **kwargs
        )

        if self.media_type == fomm.VIDEO:
            self._frames = fofr.Frames(self)
        else:
            self._frames = None

    def __repr__(self):
        kwargs = {}
        if self.media_type == fomm.VIDEO:
            kwargs["frames"] = self._frames

        return self._doc.fancy_repr(
            class_name=self.__class__.__name__, **kwargs
        )

    def _reload_backing_doc(self):
        if not self._in_db:
            return

        d = self._dataset._sample_collection.find_one({"_id": self._id})
        self._doc = self._dataset._sample_dict_to_doc(d)

    def reload(self, hard=False):
        """Reloads the sample from the database.

        Args:
            hard (False): whether to reload the sample's schema in addition to
                its field values. This is necessary if new fields may have been
                added to the dataset schema
        """
        if self.media_type == fomm.VIDEO:
            self.frames.reload(hard=hard)

        super().reload(hard=hard)

    def save(self):
        """Saves the sample to the database."""
        super().save()

    def _save(self, deferred=False):
        if not self._in_db:
            raise ValueError(
                "Cannot save a sample that has not been added to a dataset"
            )

        if self.media_type == fomm.VIDEO:
            frame_ops = self.frames._save(deferred=deferred)
        else:
            frame_ops = []

        sample_ops = super()._save(deferred=deferred)

        return sample_ops, frame_ops

    @classmethod
    def from_frame(cls, frame, filepath=None):
        """Creates a sample from the given frame.

        Args:
            frame: a :class:`fiftyone.core.frame.Frame`
            filepath (None): the path to the corresponding image frame on disk,
                if not available

        Returns:
            a :class:`Sample`
        """
        kwargs = {k: v for k, v in frame.iter_fields()}
        if filepath is not None:
            kwargs["filepath"] = filepath

        return cls(**kwargs)

    @classmethod
    def from_doc(cls, doc, dataset=None):
        """Creates a sample backed by the given document.

        Args:
            doc: a :class:`fiftyone.core.odm.sample.DatasetSampleDocument` or
                :class:`fiftyone.core.odm.sample.NoDatasetSampleDocument`
            dataset (None): the :class:`fiftyone.core.dataset.Dataset` that
                the sample belongs to

        Returns:
            a :class:`Sample`
        """
        sample = super().from_doc(doc, dataset=dataset)

        if sample.media_type == fomm.VIDEO:
            sample._frames = fofr.Frames(sample)

        return sample

    @classmethod
    def from_dict(cls, d):
        """Loads the sample from a JSON dictionary.

        The returned sample will not belong to a dataset.

        Returns:
            a :class:`Sample`
        """
        d.pop("_dataset_id", None)

        media_type = d.pop("_media_type", None)
        if media_type is None:
            media_type = fomm.get_media_type(d.get("filepath", ""))

        if media_type == fomm.VIDEO:
            frames = d.pop("frames", {})

        sample = super().from_dict(d)

        if sample.media_type == fomm.VIDEO:
            for fn, fd in frames.items():
                sample.frames[int(fn)] = fofr.Frame.from_dict(fd)

        return sampleclass Sample(_SampleMixin, Document, metaclass=SampleSingleton):
    """A sample in a :class:`fiftyone.core.dataset.Dataset`.

    Samples store all information associated with a particular piece of data in
    a dataset, including basic metadata about the data, one or more sets of
    labels (ground truth, user-provided, or FiftyOne-generated), and additional
    features associated with subsets of the data and/or label sets.

    .. note::

        :class:`Sample` instances that are **in datasets** are singletons,
        i.e.,  ``dataset[sample_id]`` will always return the same
        :class:`Sample` instance.

    Args:
        filepath: the path to the data on disk. The path is converted to an
            absolute path (if necessary) via
            :func:`fiftyone.core.storage.normalize_path`
        tags (None): a list of tags for the sample
        metadata (None): a :class:`fiftyone.core.metadata.Metadata` instance
        **kwargs: additional fields to dynamically set on the sample
    """

    _NO_DATASET_DOC_CLS = foo.NoDatasetSampleDocument

    def __init__(self, filepath, tags=None, metadata=None, **kwargs):
        super().__init__(
            filepath=filepath, tags=tags, metadata=metadata, **kwargs
        )

        if self.media_type == fomm.VIDEO:
            self._frames = fofr.Frames(self)
        else:
            self._frames = None

    def __repr__(self):
        kwargs = {}
        if self.media_type == fomm.VIDEO:
            kwargs["frames"] = self._frames

        return self._doc.fancy_repr(
            class_name=self.__class__.__name__, **kwargs
        )

    def _reload_backing_doc(self):
        if not self._in_db:
            return

        d = self._dataset._sample_collection.find_one({"_id": self._id})
        self._doc = self._dataset._sample_dict_to_doc(d)

    def reload(self, hard=False):
        """Reloads the sample from the database.

        Args:
            hard (False): whether to reload the sample's schema in addition to
                its field values. This is necessary if new fields may have been
                added to the dataset schema
        """
        if self.media_type == fomm.VIDEO:
            self.frames.reload(hard=hard)

        super().reload(hard=hard)

    def save(self):
        """Saves the sample to the database."""
        super().save()

    def _save(self, deferred=False):
        if not self._in_db:
            raise ValueError(
                "Cannot save a sample that has not been added to a dataset"
            )

        if self.media_type == fomm.VIDEO:
            frame_ops = self.frames._save(deferred=deferred)
        else:
            frame_ops = []

        sample_ops = super()._save(deferred=deferred)

        return sample_ops, frame_ops

    @classmethod
    def from_frame(cls, frame, filepath=None):
        """Creates a sample from the given frame.

        Args:
            frame: a :class:`fiftyone.core.frame.Frame`
            filepath (None): the path to the corresponding image frame on disk,
                if not available

        Returns:
            a :class:`Sample`
        """
        kwargs = {k: v for k, v in frame.iter_fields()}
        if filepath is not None:
            kwargs["filepath"] = filepath

        return cls(**kwargs)

    @classmethod
    def from_doc(cls, doc, dataset=None):
        """Creates a sample backed by the given document.

        Args:
            doc: a :class:`fiftyone.core.odm.sample.DatasetSampleDocument` or
                :class:`fiftyone.core.odm.sample.NoDatasetSampleDocument`
            dataset (None): the :class:`fiftyone.core.dataset.Dataset` that
                the sample belongs to

        Returns:
            a :class:`Sample`
        """
        sample = super().from_doc(doc, dataset=dataset)

        if sample.media_type == fomm.VIDEO:
            sample._frames = fofr.Frames(sample)

        return sample

    @classmethod
    def from_dict(cls, d):
        """Loads the sample from a JSON dictionary.

        The returned sample will not belong to a dataset.

        Returns:
            a :class:`Sample`
        """
        d.pop("_dataset_id", None)

        media_type = d.pop("_media_type", None)
        if media_type is None:
            media_type = fomm.get_media_type(d.get("filepath", ""))

        if media_type == fomm.VIDEO:
            frames = d.pop("frames", {})

        sample = super().from_dict(d)

        if sample.media_type == fomm.VIDEO:
            for fn, fd in frames.items():
                sample.frames[int(fn)] = fofr.Frame.from_dict(fd)

        return sample