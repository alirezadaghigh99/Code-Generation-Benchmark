def rec_metric_value_test_launcher(
    target_clazz: Type[RecMetric],
    target_compute_mode: RecComputeMode,
    test_clazz: Type[TestMetric],
    metric_name: str,
    task_names: List[str],
    fused_update_limit: int,
    compute_on_all_ranks: bool,
    should_validate_update: bool,
    world_size: int,
    entry_point: Callable[..., None],
    batch_window_size: int = BATCH_WINDOW_SIZE,
    test_nsteps: int = 1,
    n_classes: Optional[int] = None,
    zero_weights: bool = False,
    **kwargs: Any,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lc = get_launch_config(
            world_size=world_size, rdzv_endpoint=os.path.join(tmpdir, "rdzv")
        )

        # Call the same helper as the actual test to make code coverage visible to
        # the testing system.
        rec_metric_value_test_helper(
            target_clazz,
            target_compute_mode,
            test_clazz=None,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=compute_on_all_ranks,
            should_validate_update=should_validate_update,
            world_size=1,
            my_rank=0,
            task_names=task_names,
            batch_size=32,
            nsteps=test_nsteps,
            batch_window_size=1,
            n_classes=n_classes,
            zero_weights=zero_weights,
            **kwargs,
        )

        pet.elastic_launch(lc, entrypoint=entry_point)(
            target_clazz,
            target_compute_mode,
            task_names,
            test_clazz,
            metric_name,
            fused_update_limit,
            compute_on_all_ranks,
            should_validate_update,
            batch_window_size,
            n_classes,
            test_nsteps,
            zero_weights,
        )def get_launch_config(world_size: int, rdzv_endpoint: str) -> pet.LaunchConfig:
    return pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=world_size,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint=rdzv_endpoint,
        rdzv_configs={"store_type": "file"},
        start_method="spawn",
        monitor_interval=1,
        max_restarts=0,
    )def rec_metric_gpu_sync_test_launcher(
    target_clazz: Type[RecMetric],
    target_compute_mode: RecComputeMode,
    test_clazz: Optional[Type[TestMetric]],
    metric_name: str,
    task_names: List[str],
    fused_update_limit: int,
    compute_on_all_ranks: bool,
    should_validate_update: bool,
    world_size: int,
    entry_point: Callable[..., None],
    batch_size: int = BATCH_SIZE,
    batch_window_size: int = BATCH_WINDOW_SIZE,
    **kwargs: Any,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lc = get_launch_config(
            world_size=world_size, rdzv_endpoint=os.path.join(tmpdir, "rdzv")
        )

        # launch using torch elastic, launches for each rank
        pet.elastic_launch(lc, entrypoint=entry_point)(
            target_clazz,
            target_compute_mode,
            test_clazz,
            task_names,
            metric_name,
            world_size,
            fused_update_limit,
            compute_on_all_ranks,
            should_validate_update,
            batch_size,
            batch_window_size,
        )def rec_metric_value_test_launcher(
    target_clazz: Type[RecMetric],
    target_compute_mode: RecComputeMode,
    test_clazz: Type[TestMetric],
    metric_name: str,
    task_names: List[str],
    fused_update_limit: int,
    compute_on_all_ranks: bool,
    should_validate_update: bool,
    world_size: int,
    entry_point: Callable[..., None],
    batch_window_size: int = BATCH_WINDOW_SIZE,
    test_nsteps: int = 1,
    n_classes: Optional[int] = None,
    zero_weights: bool = False,
    **kwargs: Any,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lc = get_launch_config(
            world_size=world_size, rdzv_endpoint=os.path.join(tmpdir, "rdzv")
        )

        # Call the same helper as the actual test to make code coverage visible to
        # the testing system.
        rec_metric_value_test_helper(
            target_clazz,
            target_compute_mode,
            test_clazz=None,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=compute_on_all_ranks,
            should_validate_update=should_validate_update,
            world_size=1,
            my_rank=0,
            task_names=task_names,
            batch_size=32,
            nsteps=test_nsteps,
            batch_window_size=1,
            n_classes=n_classes,
            zero_weights=zero_weights,
            **kwargs,
        )

        pet.elastic_launch(lc, entrypoint=entry_point)(
            target_clazz,
            target_compute_mode,
            task_names,
            test_clazz,
            metric_name,
            fused_update_limit,
            compute_on_all_ranks,
            should_validate_update,
            batch_window_size,
            n_classes,
            test_nsteps,
            zero_weights,
        )