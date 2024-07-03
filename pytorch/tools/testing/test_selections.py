def calculate_shards(
    num_shards: int,
    tests: Sequence[TestRun],
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]] | None,
    must_serial: Callable[[str], bool] | None = None,
    sort_by_time: bool = True,
) -> list[tuple[float, list[ShardedTest]]]:
    must_serial = must_serial or (lambda x: True)
    test_class_times = test_class_times or {}

    # Divide tests into pytest shards
    if sort_by_time:
        known_tests = [
            x
            for x in tests
            if get_duration(x, test_file_times, test_class_times) is not None
        ]
        unknown_tests = [x for x in tests if x not in known_tests]

        pytest_sharded_tests = sorted(
            get_with_pytest_shard(known_tests, test_file_times, test_class_times),
            key=lambda j: j.get_time(),
            reverse=True,
        ) + get_with_pytest_shard(unknown_tests, test_file_times, test_class_times)
    else:
        pytest_sharded_tests = get_with_pytest_shard(
            tests, test_file_times, test_class_times
        )
    del tests

    serial_tests = [test for test in pytest_sharded_tests if must_serial(test.name)]
    parallel_tests = [test for test in pytest_sharded_tests if test not in serial_tests]

    serial_time = sum(test.get_time() for test in serial_tests)
    parallel_time = sum(test.get_time() for test in parallel_tests)
    total_time = serial_time + parallel_time / NUM_PROCS_FOR_SHARDING_CALC
    estimated_time_per_shard = total_time / num_shards
    # Separate serial tests from parallel tests as much as possible to maximize
    # parallelism by putting all the serial tests on the first num_serial_shards
    # shards. The estimated_time_limit is the estimated time it should take for
    # the least filled serial shard. Ex if we have 8 min of serial tests, 20 min
    # of parallel tests, 6 shards, and 2 procs per machine, we would expect each
    # machine to take 3 min and should aim for 3 serial shards, with shards 1
    # and 2 taking 3 min and shard 3 taking 2 min.  The estimated time limit
    # would be 2 min. This ensures that the first few shard contains as many
    # serial tests as possible and as few parallel tests as possible. The least
    # filled/last (in the example, the 3rd) shard may contain a lot of both
    # serial and parallel tests.
    estimated_time_limit = 0.0
    if estimated_time_per_shard != 0:
        estimated_time_limit = serial_time % estimated_time_per_shard
    if estimated_time_limit <= 0.01:
        estimated_time_limit = estimated_time_per_shard
    if total_time == 0:
        num_serial_shards = num_shards
    else:
        num_serial_shards = max(math.ceil(serial_time / total_time * num_shards), 1)

    sharded_jobs = [ShardJob() for _ in range(num_shards)]
    shard(
        sharded_jobs=sharded_jobs[:num_serial_shards],
        pytest_sharded_tests=serial_tests,
        estimated_time_limit=estimated_time_limit,
        serial=True,
    )
    shard(
        sharded_jobs=sharded_jobs,
        pytest_sharded_tests=parallel_tests,
        serial=False,
    )

    return [job.convert_to_tuple() for job in sharded_jobs]