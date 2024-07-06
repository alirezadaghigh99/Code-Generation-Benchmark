def build_finetuning_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: Union[int, float],
    dataset: Dict[str, Any],
    num_workers: int,
    drop_last: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    timeout: int = 0,
) -> DataSpec:
    """Builds a finetuning dataloader for training or evaluating.

    The underlying dataset can be built through one of two code paths:
        1. As a HuggingFace dataset, via `datasets.load_dataset(...)`
        2. As a streaming dataset
    You will need to set slightly different dataset config fields depending
    on which you intend to use, as explained below.

    Args:
        name (str): The type of dataloader to build. Must = "finetuning".
        ---
        *** HuggingFace dataset config fields ***
            dataset.hf_name (str, optional): The name of the HuggingFace dataset
                to use. Can also be a remote http(s) directory or object store bucket
                containing the file {split}.jsonl in the format (prompt, response),
                in which case the builder will create a HuggingFace dataset.
            dataset.hf_kwargs (DictConfig, optional): Additional kwargs to
                pass to `datasets.load_dataset`, which can be used to load
                a dataset from local files.
            dataset.preprocessing_fn (str, optional): The name/import path of
                the preprocessing function to use for formatting the data examples.
                If ``None`` (default), the builder will use the preprocessing function
                    registered under `hf_name` (see `tasks.py`), if one exists,
                    otherwise it will skip preprocessing.
                If `preprocessing_fn` corresponds to a registered preprocessing
                    function in `tasks.py`, the builder will use that.
                Otherwise, it will interpret `preprocessing_fn` as a
                    "import.path:function_name" import path; e.g., it will call
                    `from import.path import function_name` and use the imported
                    function as the preprocessing function.
            *** Streaming dataset config fields ***
            dataset.remote (str, optional): Location of a MDS-formatted
                streaming dataset to use. Setting this will tell the builder
                to create a streaming dataset rather than a HuggingFace dataset.
            dataset.local (str, optional): Local path where remote data
                will be streamed to. Only valid if `cfg.dataset.remote` has
                also been set.
            *** Shared dataset configs fields ***
            dataset.max_seq_len (int): The maximum length of sequences
                in the batch. See :class:`Seq2SeqFinetuningCollator` docstring
                for details.
            dataset.decoder_only_format (bool): Whether to format the
                examples for a decoder-only model. See :class:`Seq2SeqFinetuningCollator`
                docstring for details.
            dataset.target_responses (str): Which responses are used as training targets.
                Defaults to "last", meaning only the final response in multi-turn examples
                will serve as training targets. See :class:`Seq2SeqFinetuningCollator` docstring for
                details.
            dataset.target_prompts (str): Which prompts are used as training targets.
                Defaults to "none", meaning prompts are never used as training targets.
                See :class:`Seq2SeqFinetuningCollator` docstring for details.
            dataset.allow_pad_trimming (bool, optional): Whether to allow
                the collator to trim padding. See :class:`Seq2SeqFinetuningCollator`
                docstring for details. Default: ``False``.
            dataset.packing_ratio (Optional[float, Literal['auto']]): If provided, this invokes
                a collator wrapper that packs device_batch_size*packing_ratio
                raw examples into device_batch_size packed examples. This helps
                minimize padding while preserving sequence integrity.
                This adds `sequence_id` to the batch, which indicates which unique
                sequence each token belongs to.

                If set to 'auto', packing_ratio is profiled and the highest observed packing ratio with
                zero waste is selected.
                In practice, this may result in > 0 waste because profiling is done on only a portion
                of the dataset.

                Note: Using this feature will not change device_batch_size but it
                    will determine the number of raw examples consumed by the dataloader
                    per batch. Some examples may be discarded if they do not fit when
                    packing.
                    Select packing_ratio **carefully** based on the dataset
                    statistics, max_seq_len, and tolerance for discarding samples!
                    The script `scripts/misc/profile_packing.py` can help
                    you choose the best packing_ratio.
            dataset.shuffle (bool): Whether to shuffle the dataset.
            ___
            See :class:`StreamingFinetuningDataset` for info on other standard config
                options within `dataset` that will be passed as kwargs if
                using the streaming codepath.
            ---
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to
            prepare the data from raw text. Any missing sentinel tokens will
            be added by the collator.
        device_batch_size (int, float): The size of the batches (number of examples)
            that the dataloader will produce.
        See :class:`DataLoader` for standard argument options to the pytorch
            dataloader, such as `drop_last`, `num_workers`, etc.

    Returns:
        A pytorch dataloader

    Note:
        You can run the script inside `scripts/misc/profile_packing.py` to quickly test the
        padding/waste rates for different `cfg.dataset.packing_ratio` choices,
        given a starting workload YAML.
    """
    dataset_cfg = dataset
    _validate_config(**dataset_cfg)

    # Use EOS as the pad token if none exists
    if tokenizer.pad_token is None:  # type: ignore (sometimes it's none and that's ok)
        tokenizer.pad_token = tokenizer.eos_token

    # this full config is necessary for properly profiling the packing ratio
    dataloader_cfg = {
        'name': 'finetuning',
        'dataset': dataset_cfg,
        'drop_last': drop_last,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers,
        'timeout': timeout,
    }

    replication_factor, dataset_batch_size = construct_from_registry(
        name='dataset_replication_validator',
        registry=registry.dataset_replication_validators,
        partial_function=False,
        kwargs={
            'dataset_cfg': dataset_cfg,
            'tokenizer': tokenizer,
            'device_batch_size': device_batch_size,
        },
    )

    collate_fn, dataloader_batch_size = construct_from_registry(
        name='finetuning_collator',
        registry=registry.collators,
        partial_function=False,
        kwargs={
            'dataloader_cfg': dataloader_cfg,
            'tokenizer': tokenizer,
            'dataset_batch_size': dataset_batch_size,
        },
    )

    streaming_dataset = None  # for pyright
    sampler = None
    if dataset_cfg.get(
        'remote',
    ) is not None or dataset_cfg.get('streams') is not None:
        # Build streaming dataloader
        streams_cfg = dataset_cfg.get('streams', None)
        streams_cfg = to_dict_container(
            streams_cfg,
        ) if streams_cfg is not None else None
        streams = build_streams(
            streams_cfg,
        ) if streams_cfg is not None else None

        # note: we don't need to use ** here because we're setting default values for almost all arguments
        streaming_dataset = dataset_constructor.build_from_streaming(
            tokenizer=tokenizer,
            streams=streams,
            local=dataset_cfg.get('local', None),
            remote=dataset_cfg.get('remote', None),
            split=dataset_cfg.get('split', None),
            download_retry=dataset_cfg.get('download_retry', 2),
            download_timeout=dataset_cfg.get('download_timeout', 60),
            validate_hash=dataset_cfg.get('validate_hash', None),
            keep_zip=dataset_cfg.get('keep_zip', False),
            epoch_size=dataset_cfg.get('epoch_size', None),
            predownload=dataset_cfg.get('predownload', None),
            cache_limit=dataset_cfg.get('cache_limit', None),
            partition_algo=dataset_cfg.get('partition_algo', 'relaxed'),
            num_canonical_nodes=dataset_cfg.get('num_canonical_nodes', None),
            batch_size=dataloader_batch_size,
            shuffle=dataset_cfg.get('shuffle', False),
            shuffle_algo=dataset_cfg.get('shuffle_algo', 'py1e'),
            shuffle_seed=dataset_cfg.get('shuffle_seed', 9176),
            shuffle_block_size=dataset_cfg.get('shuffle_block_size', None),
            sampling_method=dataset_cfg.get('sampling_method', 'balanced'),
            sampling_granularity=dataset_cfg.get('sampling_granularity', 1),
            batching_method=dataset_cfg.get('batching_method', 'random'),
            max_seq_len=dataset_cfg['max_seq_len'],
            allow_unsafe_types=dataset_cfg.get('allow_unsafe_types', False),
            replication=replication_factor,
            packing_ratio=dataloader_batch_size / dataset_batch_size,
        )

    else:
        # Build HF dataloader
        dataset_name_or_path = dataset_cfg['hf_name']
        split = dataset_cfg.get('split')
        if split is None:
            raise MissingHuggingFaceURLSplitError()

        # If dataset is a remote path, download it first.
        backend, _, _ = parse_uri(dataset_name_or_path)
        if backend not in ['', None]:
            dataset_name_or_path = _download_remote_hf_dataset(
                remote_path=dataset_name_or_path,
                split=split,
            )
            split = split.replace('-', '_')

        # Get the preprocessing function.
        proto_preprocessing_fn = dataset_cfg.get('preprocessing_fn')
        if isinstance(proto_preprocessing_fn, (dict, DictConfig)):
            preprocessing_fn = dataset_constructor.get_preprocessing_fn_from_dict(
                dict(proto_preprocessing_fn),
            )
        else:
            preprocessing_fn = dataset_constructor.get_preprocessing_fn_from_str(
                proto_preprocessing_fn,
                dataset_name_or_path,
            )

        # Build dataset from HF.
        streaming_dataset = dataset_constructor.build_from_hf(
            dataset_name=dataset_name_or_path,
            split=split,
            safe_load=dataset_cfg.get('safe_load', False),
            max_seq_len=dataset_cfg['max_seq_len'],
            preprocessing_fn=preprocessing_fn,
            tokenizer=tokenizer,
            target_prompts=dataset_cfg.get(
                'target_prompts',
                _DEFAULT_TARGET_PROMPTS,
            ),
            target_responses=dataset_cfg.get(
                'target_responses',
                _DEFAULT_TARGET_RESPONSES,
            ),
            decoder_only_format=dataset_cfg['decoder_only_format'],
            hf_kwargs=dataset_cfg.get('hf_kwargs', {}),
        )

        # Ensure dataset is large enough.
        if drop_last:
            world_size = dist.get_world_size() // replication_factor
            minimum_dataset_size = world_size * dataloader_batch_size
            if hasattr(streaming_dataset, '__len__'):
                full_dataset_size = len(streaming_dataset)
                if full_dataset_size < minimum_dataset_size:
                    raise NotEnoughDatasetSamplesError(
                        dataset_name=dataset_cfg['hf_name'],
                        split=split,
                        dataloader_batch_size=dataloader_batch_size,
                        world_size=world_size,
                        full_dataset_size=full_dataset_size,
                        minimum_dataset_size=minimum_dataset_size,
                    )

        # Initialize sampler.
        sampler = dist.get_sampler(
            streaming_dataset,
            drop_last=drop_last,
            shuffle=dataset_cfg['shuffle'],
            num_replicas=dist.get_world_size() //
            replication_factor if replication_factor > 1 else None,
            rank=dist.get_global_rank() //
            replication_factor if replication_factor > 1 else None,
        )

    assert streaming_dataset is not None  # for pyright
    dl = DataLoader(
        streaming_dataset,
        collate_fn=collate_fn,
        batch_size=dataloader_batch_size,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        timeout=timeout,
    )

    return construct_from_registry(
        name='data_spec',
        registry=registry.data_specs,
        partial_function=False,
        kwargs={
            'dl': dl,
            'dataset_cfg': dataset_cfg,
        },
    )

