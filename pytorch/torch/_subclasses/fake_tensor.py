def cache_info(cls) -> DispatchCacheInfo:
        """
        Query the state of the dispatch cache.
        """
        return DispatchCacheInfo(
            FakeTensorMode.cache_hits,
            FakeTensorMode.cache_misses,
            dict(FakeTensorMode.cache_bypasses),
            len(FakeTensorMode.cache),
        )

