def _get_threadpool_controller():
    """Return the global threadpool controller instance."""
    global _threadpool_controller

    if _threadpool_controller is None:
        _threadpool_controller = ThreadpoolController()

    return _threadpool_controller

class Parallel(joblib.Parallel):
    """Tweak of :class:`joblib.Parallel` that propagates the scikit-learn configuration.

    This subclass of :class:`joblib.Parallel` ensures that the active configuration
    (thread-local) of scikit-learn is propagated to the parallel workers for the
    duration of the execution of the parallel tasks.

    The API does not change and you can refer to :class:`joblib.Parallel`
    documentation for more details.

    .. versionadded:: 1.3
    """

    def __call__(self, iterable):
        """Dispatch the tasks and return the results.

        Parameters
        ----------
        iterable : iterable
            Iterable containing tuples of (delayed_function, args, kwargs) that should
            be consumed.

        Returns
        -------
        results : list
            List of results of the tasks.
        """
        # Capture the thread-local scikit-learn configuration at the time
        # Parallel.__call__ is issued since the tasks can be dispatched
        # in a different thread depending on the backend and on the value of
        # pre_dispatch and n_jobs.
        config = get_config()
        iterable_with_config = (
            (_with_config(delayed_func, config), args, kwargs)
            for delayed_func, args, kwargs in iterable
        )
        return super().__call__(iterable_with_config)

