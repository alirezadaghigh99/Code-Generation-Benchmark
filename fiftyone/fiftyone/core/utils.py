class StaticBatcher(Batcher):
    """Class for iterating over the elements of an iterable with a static
    batch size.

    This class is often used in conjunction with a :class:`ProgressBar` to keep
    the user appraised on the status of a long-running task.

    Example usage::

        import fiftyone.core.utils as fou

        elements = range(int(1e7))

        batcher = fou.StaticBatcher(elements, batch_size=10000)

        for batch in batcher:
            print("batch size: %d" % len(batch))

        batcher = fou.StaticBatcher(elements, batch_size=10000, progress=True)

        with batcher:
            for batch in batcher:
                print("batch size: %d" % len(batch))

    Args:
        iterable: an iterable to batch over. If ``None``, the result of
            ``next()`` will be a batch size instead of a batch, and is an
            infinite iterator.
        batch_size: size of batches to generate
        return_views (False): whether to return each batch as a
            :class:`fiftyone.core.view.DatasetView`. Only applicable when the
            iterable is a :class:`fiftyone.core.collections.SampleCollection`
        progress (False): whether to render a progress bar tracking the
            consumption of the batches (True/False), use the default value
            ``fiftyone.config.show_progress_bars`` (None), or a progress
            callback function to invoke instead
        total (None): the length of ``iterable``. Only applicable when
            ``progress=True``. If not provided, it is computed via
            ``len(iterable)``, if possible
    """

    def __init__(
        self,
        iterable,
        batch_size,
        return_views=False,
        progress=False,
        total=None,
    ):
        super().__init__(
            iterable, return_views=return_views, progress=progress, total=total
        )

        self.batch_size = batch_size

    def _compute_batch_size(self):
        return self.batch_size