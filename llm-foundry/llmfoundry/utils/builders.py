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
        kwargs['train_config'] = copy.deepcopy(train_config)
        registry_to_use = registry.callbacks_with_config

    return construct_from_registry(
        name=name,
        registry=registry_to_use,
        partial_function=True,
        pre_validation_function=Callback,
        post_validation_function=None,
        kwargs=kwargs,
    )

def build_tokenizer(
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

    return tokenizer

def build_composer_model(
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

def build_icl_evaluators(
    icl_tasks: Union[str, List[Dict[str, Any]]],
    tokenizer: PreTrainedTokenizerBase,
    default_max_seq_len: int,
    default_batch_size: int,
    destination_dir: Optional[str] = None,
    icl_subset_num_batches: Optional[int] = None,
) -> Tuple[List[Evaluator], List[str]]:
    if destination_dir is None:
        destination_dir = os.getcwd()

    evaluators = []
    logger_keys = []

    icl_tasks_list = None
    if isinstance(icl_tasks, str):
        log.info(f'Extracting ICL task config from path: {icl_tasks}')
        with open(icl_tasks, 'r') as icl_f:
            icl_task_cfg = om.load(icl_f)
        icl_tasks_list = to_list_container(icl_task_cfg.icl_tasks)
    else:
        icl_tasks_list = icl_tasks

    def _validate_cfg(icl_cfg: Dict[str, Any]):
        assert 'label' in icl_cfg
        assert 'dataset_uri' in icl_cfg and icl_cfg['dataset_uri'] is not None
        assert 'icl_task_type' in icl_cfg
        assert 'num_fewshot' in icl_cfg

        if 'metric_names' not in icl_cfg:
            if icl_cfg['icl_task_type'] == 'language_modeling':
                icl_cfg['metric_names'] = ['InContextLearningLMAccuracy']
            elif icl_cfg['icl_task_type'] == 'multiple_choice':
                icl_cfg['metric_names'] = [
                    'InContextLearningMultipleChoiceAccuracy',
                ]
            elif icl_cfg['icl_task_type'] == 'schema':
                icl_cfg['metric_names'] = [
                    'InContextLearningMultipleChoiceAccuracy',
                ]
            elif icl_cfg['icl_task_type'] == 'generation_task_with_answers':
                icl_cfg['metric_names'] = [
                    'InContextLearningGenerationExactMatchAccuracy',
                ]
            else:
                raise ValueError(
                    f'No metric_names defined, unable to build default metrics for icl_task_type={icl_cfg["icl_task_type"]}.',
                )

        if 'max_seq_len' not in icl_cfg:
            icl_cfg['max_seq_len'] = default_max_seq_len
        if 'batch_size' not in icl_cfg:
            icl_cfg['batch_size'] = default_batch_size

        if 'num_beams' in icl_cfg:
            raise ValueError(
                'num_beams is no longer supported as a top level icl_task parameter.'  + \
                'Please use generation_kwargs.num_beams instead.')

    for icl_cfg in icl_tasks_list:
        assert isinstance(
            icl_cfg,
            dict,
        ), f'Expected dict, got {type(icl_cfg)}, {icl_cfg=}'
        _validate_cfg(icl_cfg)
        for num_fewshot in list(icl_cfg['num_fewshot']):
            if tokenizer.pad_token_id is None:
                # Current workaround to support GPT2 tokenizer with `pad_token_id = None`
                pad_tok_id = tokenizer.eos_token_id
            else:
                pad_tok_id = tokenizer.pad_token_id

            label = f'{icl_cfg["label"]}/{num_fewshot}-shot'
            metric_names = list(icl_cfg['metric_names'])
            # TODO: fix Composer bug when copying local paths and destination exists
            destination_path = f'{destination_dir}/{icl_cfg["label"]}-{num_fewshot}.jsonl'
            if dist.get_local_rank() == 0 and os.path.exists(destination_path):
                os.remove(destination_path)
            dist.barrier()

            hf_parsing_map = icl_cfg.get('hf_parsing_map', {})
            hf_loading_vars = icl_cfg.get('hf_loading_vars', {})
            early_stopping_criteria = icl_cfg.get(
                'early_stopping_criteria',
                [],
            )
            # TODO: fix manual removal of non-constructor fields
            icl_constructor_kwargs = copy.deepcopy(icl_cfg)
            icl_constructor_kwargs.pop('label', None)
            icl_constructor_kwargs.pop('metric_names', None)
            icl_constructor_kwargs.pop('icl_task_type', None)
            icl_constructor_kwargs.pop('batch_size', None)
            icl_constructor_kwargs.pop('has_categories', None)

            # Add custom constructor arguments
            icl_constructor_kwargs['pad_tok_id'] = pad_tok_id
            icl_constructor_kwargs['num_fewshot'] = num_fewshot

            # Support backwards compatibility for the naming of "prelimiter" as "question_prelimiter"
            if 'question_prelimiter' in icl_constructor_kwargs:
                if 'prelimiter' in icl_constructor_kwargs:
                    raise ValueError(
                        'Both "question_prelimiter" and "prelimiter" are specified in the ICL task config. '
                        +
                        'Please only specify one of them, as they map to the same argument.',
                    )
                else:
                    icl_constructor_kwargs['prelimiter'
                                          ] = icl_constructor_kwargs.pop(
                                              'question_prelimiter',
                                          )

            assert early_stopping_criteria is None or isinstance(
                early_stopping_criteria,
                list,
            )

            dataloaders = get_icl_task_dataloader(
                icl_task_type=icl_cfg['icl_task_type'],
                dataset_uri=icl_cfg['dataset_uri'],
                tokenizer=tokenizer,
                batch_size=icl_cfg['batch_size'],
                hf_loading_vars=hf_loading_vars,
                hf_parsing_map=hf_parsing_map,
                has_categories=icl_cfg.get('has_categories', False),
                destination_path=destination_path,
                kwargs=icl_constructor_kwargs,
            )
            if 'has_categories' in icl_cfg and icl_cfg[
                'has_categories'] and isinstance(dataloaders, dict):
                for category in dataloaders.keys():
                    logger_keys.extend([
                        f'metrics/{label}/{category}/{m}' for m in metric_names
                    ])
                    evaluators.append(
                        Evaluator(
                            label=f'{label}/{category}',
                            dataloader=dataloaders[category],
                            metric_names=metric_names,
                        ),
                    )
            else:
                logger_keys.extend([
                    f'metrics/{label}/{m}' for m in metric_names
                ])
                evaluators.append(
                    Evaluator(
                        label=label,
                        dataloader=dataloaders,
                        metric_names=metric_names,
                        subset_num_batches=icl_subset_num_batches,
                    ),
                )

    return evaluators, logger_keys

def build_optimizer(
    model: torch.nn.Module,
    name: str,
    optimizer_config: Dict[str, Any],
) -> Optimizer:

    params = _extract_param_groups(model, optimizer_config)
    kwargs = {**optimizer_config}

    if 'params' in kwargs:
        raise ValueError(
            'The `params` will be automatically extracted from the model and ' +
            'optimizer config. Please remove it from the optimizer config kwargs.',
        )

    kwargs['params'] = params
    return construct_from_registry(
        name=name,
        registry=registry.optimizers,
        partial_function=True,
        pre_validation_function=Optimizer,
        post_validation_function=None,
        kwargs=kwargs,
    )

