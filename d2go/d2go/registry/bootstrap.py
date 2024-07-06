def bootstrap_registries(enable_cache: bool = True, catch_exception: bool = True):
    """
    Bootstrap all registries so that all objects are effectively registered.

    This function will "import" all the files from certain locations (eg. d2go package)
    and look for a set of known registries (eg. d2go's builtin registries). The "import"
    should not have any side effect, which is achieved by mocking builtin.__import__.
    """

    global _IS_BOOTSTRAPPED
    if _IS_BOOTSTRAPPED:
        logger.warning("Registries are already bootstrapped, skipped!")
        return

    if _INSIDE_BOOTSTRAP:
        _log(1, "calling bootstrap_registries() inside bootstrap process, skip ...")
        return

    start = time.perf_counter()

    # load cached bootstrap results if exist
    cached_bootstrap_results: Dict[str, CachedResult] = {}
    if enable_cache:
        filename = os.path.join(_get_cache_dir(), _BOOTSTRAP_CACHE_FILENAME)
        if os.path.isfile(filename):
            logger.info(f"Loading bootstrap cache at {filename} ...")
            cached_bootstrap_results = _load_cached_results(filename)
        else:
            logger.info(
                f"Can't find the bootstrap cache at {filename}, start from scratch"
            )

    # locate all the files under d2go package
    # NOTE: we may extend to support user-defined locations if necessary
    d2go_root = pkg_resources.resource_filename("d2go", "")
    logger.info(f"Start bootstrapping for d2go_root: {d2go_root} ...")
    all_files = glob.glob(f"{d2go_root}/**/*.py", recursive=True)
    all_files = [os.path.relpath(x, os.path.dirname(d2go_root)) for x in all_files]

    new_bootstrap_results: Dict[str, CachedResult] = {}
    files_per_status = defaultdict(list)
    time_per_file = {}
    for filename in all_files:
        _log(1, f"bootstrap for file: {filename}")

        cached_result = cached_bootstrap_results.get(filename, None)
        with _catchtime() as t:
            result, status = _bootstrap_file(filename, catch_exception, cached_result)
        new_bootstrap_results[filename] = result
        files_per_status[status].append(filename)
        time_per_file[filename] = t.time

    end = time.perf_counter()
    duration = end - start
    status_breakdown = ", ".join(
        [f"{len(files_per_status[status])} {status.name}" for status in BootstrapStatus]
    )
    logger.info(
        f"Finished bootstrapping for {len(all_files)} files ({status_breakdown})"
        f" in {duration:.2f} seconds."
    )
    exception_files = [
        filename
        for filename, result in new_bootstrap_results.items()
        if result.status == BootstrapStatus.FAILED.name
    ]
    if len(exception_files) > 0:
        logger.warning(
            "Found exception for the following {} files (either during this bootstrap"
            " run or from previous cached result), registration inside those files"
            " might not work!\n{}".format(
                len(exception_files),
                "\n".join(exception_files),
            )
        )

    # Log slowest Top-N files
    TOP_N = 100
    _log(2, f"Top-{TOP_N} slowest files during bootstrap:")
    all_time = [(os.path.relpath(k, d2go_root), v) for k, v in time_per_file.items()]
    for x in sorted(all_time, key=lambda x: x[1])[-TOP_N:]:
        _log(2, x)

    if enable_cache:
        filename = os.path.join(_get_cache_dir(), _BOOTSTRAP_CACHE_FILENAME)
        logger.info(f"Writing updated bootstrap results to {filename} ...")
        _dump_cached_results(new_bootstrap_results, filename)

    _IS_BOOTSTRAPPED = True

