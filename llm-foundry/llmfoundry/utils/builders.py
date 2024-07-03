def build_callback(
    name: str,
    kwargs: Optional[Dict[str, Any]] = None,
    train_config: Any = None,
) -> Callback:
    """Builds a callback from the registry."""
    registry_to_use = registry.callbacks
    if name in registry.callbacks_with_config:
        if kwargs is None:
            kwargs = {}
        if 'train_config' in kwargs:
            raise ValueError(
                f'`train_config` is a reserved keyword for callbacks with config. Please remove it from the kwargs.',
            )
        kwargs['train_config'] = train_config
        registry_to_use = registry.callbacks_with_config

    return construct_from_registry(
        name=name,
        registry=registry_to_use,
        partial_function=True,
        pre_validation_function=Callback,
        post_validation_function=None,
        kwargs=kwargs,
    )def build_tokenizer(
    tokenizer_name: str,
    tokenizer_kwargs: Dict[str, Any],
) -> PreTrainedTokenizerBase:
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    signal_file_path = f'.node_{dist.get_node_rank()}_local_rank0_completed_tokenizer_setup'

    if dist.is_available() and dist.is_initialized(
    ) and dist.get_world_size() > 1:
        # Make sure the tokenizer files are downloaded and cached first by local rank 0
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            pass

    if tokenizer_name.startswith('tiktoken'):
        tokenizer = TiktokenTokenizerWrapper(**tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            **tokenizer_kwargs,
        )

        # HuggingFace does not respect the model_max_length kwarg, and overrides it with
        # min(kwargs['model_max_length'], original_config['model_max_length']), so we
        # explicitly set it here
        tokenizer.model_max_length = tokenizer_kwargs.get(
            'model_max_length',
            int(1e30),
        )

    if not hasattr(tokenizer, 'eos_token') or tokenizer.eos_token is None:
        raise ValueError(
            f'The tokenizer {tokenizer_name} must have an eos_token.',
        )

    if dist.is_available() and dist.is_initialized(
    ) and dist.get_world_size() > 1:
        if dist.get_local_rank() == 0:
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed_tokenizer_setup')

        dist.barrier()

        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)

    return tokenizerdef build_composer_model(
    name: str,
    cfg: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    init_context: Optional[ContextManager] = None,
    master_weights_dtype: Optional[str] = None,
) -> ComposerModel:
    """Builds a ComposerModel from the registry.

    Args:
        name (str): Name of the model to build.
        cfg (DictConfig): Configuration for the model.
        tokenizer (PreTrainedTokenizerBase): Tokenizer to use.
        init_context (Optional[ContextManager], optional): Context manager to use for initialization. Defaults to None.
        master_weights_dtype (Optional[str], optional): Master weights dtype. Defaults to None.

    Returns:
        ComposerModel: _description_
    """
    if init_context is None:
        init_context = contextlib.nullcontext()

    with init_context:
        model = construct_from_registry(
            name=name,
            registry=registry.models,
            pre_validation_function=ComposerModel,
            post_validation_function=None,
            kwargs={
                **cfg,
                'tokenizer': tokenizer,
            },
        )

    str_dtype_to_torch_dtype = {
        'f16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
    }

    if master_weights_dtype is not None:
        if master_weights_dtype not in str_dtype_to_torch_dtype:
            raise ValueError(
                f'Invalid master_weights_dtype: {master_weights_dtype}. ' +
                f'Valid options are: {list(str_dtype_to_torch_dtype.keys())}.',
            )
        dtype = str_dtype_to_torch_dtype[master_weights_dtype]
        model = model.to(dtype=dtype)

    return model