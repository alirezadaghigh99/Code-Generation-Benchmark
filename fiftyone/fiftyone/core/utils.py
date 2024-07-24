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

class ContentSizeDynamicBatcher(BaseDynamicBatcher):
    """Class for iterating over the elements of an iterable with a dynamic
    batch size to achieve a desired content size.

    The batch sizes emitted when iterating over this object are dynamically
    scaled such that the total content size of the batch is as close as
    possible to a specified target size.

    This batcher requires that backpressure feedback be provided, either by
    providing a BSON-able batch from which the content size can be computed,
    or by manually providing the content size.

    This class is often used in conjunction with a :class:`ProgressBar` to keep
    the user appraised on the status of a long-running task.

    Example usage::

        import fiftyone.core.utils as fou

        elements = range(int(1e7))

        batcher = fou.ContentSizeDynamicBatcher(
            elements, target_size=2**20, max_batch_beta=2.0
        )

        # Raises ValueError after first batch, we forgot to apply backpressure
        for batch in batcher:
            print("batch size: %d" % len(batch))

        # Now it works
        for batch in batcher:
            print("batch size: %d" % len(batch))
            batcher.apply_backpressure(batch)

        batcher = fou.ContentSizeDynamicBatcher(
            elements,
            target_size=2**20,
            max_batch_beta=2.0,
            progress=True
        )

        with batcher:
            for batch in batcher:
                print("batch size: %d" % len(batch))
                batcher.apply_backpressure(batch)

    Args:
        iterable: an iterable to batch over. If ``None``, the result of
            ``next()`` will be a batch size instead of a batch, and is an
            infinite iterator.
        target_size (1048576): the target batch bson content size, in bytes
        init_batch_size (1): the initial batch size to use
        min_batch_size (1): the minimum allowed batch size
        max_batch_size (None): an optional maximum allowed batch size
        max_batch_beta (None): an optional lower/upper bound on the ratio
            between successive batch sizes
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

    manual_backpressure = True

    def __init__(
        self,
        iterable,
        target_size=2**20,
        init_batch_size=1,
        min_batch_size=1,
        max_batch_size=None,
        max_batch_beta=None,
        return_views=False,
        progress=False,
        total=None,
    ):
        super().__init__(
            iterable,
            target_size,
            init_batch_size=init_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            max_batch_beta=max_batch_beta,
            return_views=return_views,
            progress=progress,
            total=total,
        )
        self._last_batch_content_size = 0

    def apply_backpressure(self, batch_or_size):
        if isinstance(batch_or_size, numbers.Number):
            batch_content_size = batch_or_size
        else:
            batch_content_size = sum(
                len(json_util.dumps(obj)) for obj in batch_or_size
            )

        self._last_batch_content_size = batch_content_size
        self._manually_applied_backpressure = True

    def _get_measurement(self):
        return self._last_batch_content_size

class ProgressBar(etau.ProgressBar):
    """.. autoclass:: eta.core.utils.ProgressBar"""

    def __init__(self, total=None, progress=None, quiet=None, **kwargs):
        if progress is None:
            progress = fo.config.show_progress_bars

        if quiet is not None:
            progress = not quiet

        if callable(progress):
            callback = progress
            progress = False
        else:
            callback = None

        kwargs["total"] = total
        if isinstance(progress, bool):
            kwargs["quiet"] = not progress

        if "iters_str" not in kwargs:
            kwargs["iters_str"] = "samples"

        # For progress bars in notebooks, use a fixed size so that they will
        # read well across browsers, in HTML format, etc
        if foc.is_notebook_context() and "max_width" not in kwargs:
            kwargs["max_width"] = 90

        super().__init__(**kwargs)

        self._progress = progress
        self._callback = callback

    def __call__(self, iterable):
        # Ensure that `len(iterable)` is not computed unnecessarily
        no_len = self._quiet and self._total is None
        if no_len:
            self._total = -1

        super().__call__(iterable)

        if no_len:
            self._total = None

        return self

    def set_iteration(self, *args, **kwargs):
        super().set_iteration(*args, **kwargs)

        if self._callback is not None:
            self._callback(self)

class SetAttributes(object):
    """Context manager that temporarily sets the attributes of a class to new
    values.

    Args:
        obj: the object
        **kwargs: the attribute key-values to set while the context is active
    """

    def __init__(self, obj, **kwargs):
        self._obj = obj
        self._kwargs = kwargs
        self._orig_kwargs = None

    def __enter__(self):
        self._orig_kwargs = {}
        for k, v in self._kwargs.items():
            self._orig_kwargs[k] = getattr(self._obj, k)
            setattr(self._obj, k, v)

        return self

    def __exit__(self, *args):
        for k, v in self._orig_kwargs.items():
            setattr(self._obj, k, v)

