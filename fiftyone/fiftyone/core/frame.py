class Frame(Document, metaclass=FrameSingleton):
    """A frame in a video :class:`fiftyone.core.sample.Sample`.

    Frames store all information associated with a particular frame of a video,
    including one or more sets of labels (ground truth, user-provided, or
    FiftyOne-generated) as well as additional features associated with subsets
    of the data and/or label sets.

    .. note::

        :class:`Frame` instances that are attached to samples **in datasets**
        are singletons, i.e.,  ``sample.frames[frame_number]`` will always
        return the same :class:`Frame` instance.

    Args:
        **kwargs: frame fields and values
    """

    _NO_DATASET_DOC_CLS = foo.NoDatasetFrameDocument

    @property
    def dataset_id(self):
        return self._doc._dataset_id

    @property
    def _dataset_id(self):
        _id = self._doc._dataset_id
        return ObjectId(_id) if _id is not None else None

    @property
    def sample_id(self):
        return self._doc._sample_id

    @property
    def _sample_id(self):
        _id = self._doc._sample_id
        return ObjectId(_id) if _id is not None else None

    def save(self):
        """Saves the frame to the database."""
        if not self._in_db:
            raise ValueError(
                "Use `sample.save()` to save newly added frames to a sample"
            )

        super().save()

    def _reload_backing_doc(self):
        if not self._in_db:
            return

        d = self._dataset._frame_collection.find_one(
            {"_sample_id": self._sample_id, "frame_number": self.frame_number}
        )
        self._doc = self._dataset._frame_dict_to_doc(d)

