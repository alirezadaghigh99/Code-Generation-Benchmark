def _cache_transform(tape: QuantumTape, cache: MutableMapping):
    """Caches the result of ``tape`` using the provided ``cache``.

    .. note::

        This function makes use of :attr:`.QuantumTape.hash` to identify unique tapes.
    """

    def cache_hit_postprocessing(_results: Tuple[Tuple]) -> Tuple:
        result = cache[tape.hash]
        if result is not None:
            if tape.shots and getattr(cache, "_persistent_cache", True):
                warnings.warn(_CACHED_EXECUTION_WITH_FINITE_SHOTS_WARNINGS, UserWarning)
            return result

        raise RuntimeError(
            "Result for tape is missing from the execution cache. "
            "This is likely the result of a race condition."
        )

    if tape.hash in cache:
        return [], cache_hit_postprocessing

    def cache_miss_postprocessing(results: Tuple[Tuple]) -> Tuple:
        result = results[0]
        cache[tape.hash] = result
        return result

    # Adding a ``None`` entry to the cache indicates that a result will eventually be available for
    # the tape. This assumes that post-processing functions are called in the same order in which
    # the transforms are invoked. Otherwise, ``cache_hit_postprocessing()`` may be called before the
    # result of the corresponding tape is placed in the cache by ``cache_miss_postprocessing()``.
    cache[tape.hash] = None
    return [tape], cache_miss_postprocessing

def _apply_cache_transform(fn: Callable, cache: Optional[MutableMapping]) -> Callable:
    """Wraps the given execution function with ``_cache_transform()`` using the provided cache.

    Args:
        fn (Callable): The execution function to be augmented with caching. This function should
            have the signature ``fn(tapes, **kwargs)`` and return ``list[tensor_like]`` with the
            same length as the input ``tapes``.
        cache (None | MutableMapping): The cache to use. If ``None``, caching will not occur.
    """
    if cache is None:
        return fn

    def execution_function_with_caching(tapes):
        tapes, post_processing_fn = _cache_transform(tapes, cache=cache)
        return post_processing_fn(fn(tapes))

    return execution_function_with_caching

