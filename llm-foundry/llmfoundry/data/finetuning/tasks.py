def tokenize_formatted_example(
    example: Example,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenizedExample:
    """Tokenizes a formatted example using the provided tokenizer.

    Args:
        example (Example): The input example to be tokenized.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for tokenization.

    Returns:
        TokenizedExample: The tokenized example.

    Raises:
        ValueError: If the example format is unknown.
    """
    example_format = _get_example_type(example)

    if example_format == 'chat':
        chat_example = cast(ChatFormattedDict, example)
        return _tokenize_chat_formatted_example(chat_example, tokenizer)
    elif example_format == 'prompt_response':
        prompt_response_example: PromptResponseDict = cast(
            PromptResponseDict,
            example,
        )
        return _tokenize_prompt_response_formatted_example(
            prompt_response_example,
            tokenizer,
        )
    else:
        raise NotImplementedError

class StreamingFinetuningDataset(StreamingDataset):
    """Finetuning dataset with flexible tokenization using StreamingDataset.

    Args:
        tokenizer (Tokenizer): The name of the HuggingFace tokenizer to use to
            tokenize samples.
        token_encoding_type (str): The encoding type of the tokenized samples. This is only used
            for legacy datasets that have been written directly as 'bytes' instead of numpy
            arrays. Types are auto-inferred for numpy arrays. Defaults to 'int64'.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str): Local dataset directory where shards are cached by split.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. If ``None``, its value is set to ``8 * batch_size``. Defaults to ``None``.
        cache_limit (Union[int, str], optional) - Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s) may
            be evicted (deleted from the local cache) in order to stay under the limit. Set to None
            to disable shard eviction. Supports integer bytes as well as string human-readable
            bytes (e.g., 100b, 64kb, 77mb, and so on). Defaults to None.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. If ``None``, this is interpreted as 64 times the number of physical
            nodes of the initial run if ``shuffle_algo`` is ``py1s`` or ``py2s``, and simply the
            number of physical nodes of the initial run otherwise. Defaults to ``None``.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. If ``None``, its value is calculated as
            ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to ``None``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial repeat,
            how many samples to pick from the same shard at a time (``1`` for evenly balanced
            across shards, ``1000`` to pick 1000 samples from the same shard at a time, etc).
            Defaults to ``1``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``, or
            ``per_stream``. Defaults to ``random``.
        allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
            execution during deserialization, whether to keep going if ``True`` or raise an error
            if ``False``. Defaults to ``False``.
        replication (int, optional): Determines how many consecutive devices will receive the same
            samples. Useful for training with tensor or sequence parallelism, where multiple
            devices need to see the same partition of the dataset. Defaults to ``None``.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        token_encoding_type: str = 'int64',
        streams: Optional[Sequence[Stream]] = None,
        local: Optional[str] = None,
        remote: Optional[str] = None,
        split: Optional[str] = None,
        download_retry: int = 2,
        download_timeout: float = 60,
        validate_hash: Optional[str] = None,
        keep_zip: bool = False,
        epoch_size: Optional[Union[int, str]] = None,
        predownload: Optional[int] = None,
        cache_limit: Optional[Union[int, str]] = None,
        partition_algo: str = 'relaxed',
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = 'py1e',
        shuffle_seed: int = 9176,
        shuffle_block_size: Optional[int] = None,
        sampling_method: str = 'balanced',
        sampling_granularity: int = 1,
        batching_method: str = 'random',
        max_seq_len: int = 2048,
        allow_unsafe_types: bool = False,
        replication: Optional[int] = None,
        packing_ratio: Optional[float] = None,
        **kwargs: Any,
    ):

        if len(kwargs) > 0:
            raise ValueError(
                f'StreamingFinetuningDataset() got an unexpected keyword argument: {kwargs}',
            )

        if token_encoding_type not in SUPPORTED_MDS_ENCODING_TYPES:
            raise ValueError(
                f'The token_encoding_type must be one of {SUPPORTED_MDS_ENCODING_TYPES}, but got {token_encoding_type}',
            )
        self.token_encoding_type = token_encoding_type

        if streams is None:
            stream_remote_local_validate(remote, local, split)
        else:
            for stream in streams:
                stream_remote_local_validate(
                    stream.remote,
                    stream.local,
                    split,
                )

        super().__init__(
            streams=streams,
            local=local,
            remote=remote,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            cache_limit=cache_limit,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            shuffle_block_size=shuffle_block_size,
            sampling_method=sampling_method,
            sampling_granularity=sampling_granularity,
            batching_method=batching_method,
            allow_unsafe_types=allow_unsafe_types,
            replication=replication,
        )

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.packing_ratio = packing_ratio

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        if 'turns' in sample:
            # Already tokenized in latest format
            return sample
        if 'input_ids' in sample:
            # Already tokenized data (old format)
            if isinstance(sample['input_ids'], bytes):
                sample['input_ids'] = np.frombuffer(
                    sample['input_ids'],
                    dtype=getattr(np, self.token_encoding_type),
                )[:self.max_seq_len].tolist().copy()
                sample['labels'] = np.frombuffer(
                    sample['labels'],
                    dtype=getattr(np, self.token_encoding_type),
                )[:self.max_seq_len].tolist().copy()
            elif isinstance(sample['input_ids'], np.ndarray):
                sample['input_ids'] = sample['input_ids'][:self.max_seq_len
                                                         ].tolist().copy()
                sample['labels'] = sample['labels'][:self.max_seq_len].tolist(
                ).copy()
            else:
                raise ValueError(
                    f'Expect input_ids to be bytes or numpy.ndarray type, but got {type(sample["input_ids"])}',
                )
            # Convert to latest format by wrapping sample as a "turn"
            return {'turns': [sample]}
        return tokenize_formatted_example(sample, tokenizer=self.tokenizer)

    def state_dict(self, num_samples: int,
                   from_beginning: bool) -> Dict[str, Any]:
        if self.packing_ratio is not None:
            num_samples = int(self.packing_ratio * num_samples)

        return super().state_dict(
            num_samples=num_samples,
            from_beginning=from_beginning,
        )

