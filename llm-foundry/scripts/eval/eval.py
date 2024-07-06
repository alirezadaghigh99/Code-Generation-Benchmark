def main(cfg: DictConfig) -> Tuple[List[Trainer], pd.DataFrame]:
    # Run user provided code if specified
    for code_path in cfg.get('code_paths', []):
        import_file(code_path)

    logged_cfg, eval_config = make_dataclass_and_log_config(
        cfg,
        EvalConfig,
        EVAL_CONFIG_KEYS,
        icl_tasks_required=True,
    )

    model_configs = eval_config.models
    eval_gauntlet_config = eval_config.eval_gauntlet or eval_config.eval_gauntlet_str

    fsdp_config = eval_config.fsdp_config

    # Mandatory Evaluation Parameters
    icl_tasks = eval_config.icl_tasks or eval_config.icl_tasks_str
    if icl_tasks is None:
        raise ValueError('icl_tasks must be specified in the config')

    # Optional Evaluation Parameters with default values
    eval_loader_config = eval_config.eval_loader or eval_config.eval_loaders
    default_run_name: str = os.environ.get('RUN_NAME', 'llm')
    run_name = eval_config.run_name if eval_config.run_name else default_run_name

    reproducibility.seed_all(eval_config.seed)
    dist.initialize_dist(get_device(None), timeout=eval_config.dist_timeout)

    if eval_config.python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=
            f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
        )
        logging.getLogger('llmfoundry').setLevel(
            eval_config.python_log_level.upper(),
        )

    # Default argument values for evaluate_model
    eval_gauntlet_df = None
    models_df = None
    composite_scores = None
    trainers = []

    # Build loggers
    loggers: List[LoggerDestination] = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (eval_config.loggers or {}).items()
    ]

    mosaicml_logger = find_mosaicml_logger(loggers)
    if mosaicml_logger is None:
        mosaicml_logger = maybe_create_mosaicml_logger()
        # mosaicml_logger will be None if run isn't on MosaicML platform
        if mosaicml_logger is not None:
            loggers.append(mosaicml_logger)

    # mosaicml_logger will be None if the run isn't from the MosaicML platform
    if mosaicml_logger is not None:
        log_eval_analytics(
            mosaicml_logger,
            model_configs,
            icl_tasks,
            eval_gauntlet_config,
        )

    for model_cfg in model_configs:

        attn_config = model_cfg['model'].get('attn_config', None)
        if attn_config is not None:
            seq_parallel_world_size = attn_config.get(
                'seq_parallel_world_size',
                None,
            )
            if seq_parallel_world_size is not None and seq_parallel_world_size != 1:
                raise ValueError(
                    'Offline eval does not support sequence parallelism.',
                )

        (trainer, logger_keys, eval_gauntlet_callback,
         eval_gauntlet_df) = evaluate_model(
             dist_timeout=eval_config.dist_timeout,
             run_name=run_name,
             seed=eval_config.seed,
             icl_tasks=icl_tasks,
             max_seq_len=eval_config.max_seq_len,
             device_eval_batch_size=eval_config.device_eval_batch_size,
             eval_gauntlet_config=eval_gauntlet_config,
             eval_loader_config=eval_loader_config,
             fsdp_config=fsdp_config,
             loggers=loggers,
             python_log_level=eval_config.python_log_level,
             precision=eval_config.precision,
             eval_gauntlet_df=eval_gauntlet_df,
             callback_configs=eval_config.callbacks,
             eval_subset_num_batches=eval_config.eval_subset_num_batches,
             icl_subset_num_batches=eval_config.icl_subset_num_batches,
             metadata=eval_config.metadata,
             logged_config=logged_cfg,
             should_log_config=eval_config.log_config,
             **model_cfg,
         )
        trainers.append(trainer)

        if eval_gauntlet_callback is not None:
            composite_scores = eval_gauntlet_callback.eval_after_all(
                trainer.state,
                trainer.logger,
            )

        benchmark_to_taxonomy = {}
        if eval_gauntlet_callback is not None:
            for t in eval_gauntlet_callback.categories:
                for b in t['benchmarks']:
                    benchmark_to_taxonomy[b['name']] = t['name']

        assert 'model_name' in model_cfg, 'model_name must be specified in model config'
        model_results = calculate_markdown_results(
            logger_keys,
            trainer,
            benchmark_to_taxonomy,
            model_cfg['model_name'],
        )

        if models_df is None:
            models_df = model_results
        else:
            models_df = pd.concat([models_df, model_results], ignore_index=True)

        if eval_gauntlet_df is not None and eval_gauntlet_callback is not None:
            assert composite_scores is not None
            row = {'model_name': model_cfg['model_name']}
            row.update({
                k.split('/')[-1]: v for k, v in composite_scores.items()
            })
            eval_gauntlet_df = pd.concat([
                eval_gauntlet_df,
                pd.DataFrame([row]),
            ],
                                         ignore_index=True)

            print(f'Printing gauntlet results for all models')

            print(
                eval_gauntlet_df.sort_values(
                    list(eval_gauntlet_callback.averages.keys())[0],
                    ascending=False,
                ).to_markdown(index=False),
            )
        print(f'Printing complete results for all models')
        assert models_df is not None
        print(models_df.to_markdown(index=False))

        trainer.close()

    return trainers, eval_gauntlet_df

