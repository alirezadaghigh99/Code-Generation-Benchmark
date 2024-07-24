def register_default_init_args(
    nncf_config: NNCFConfig,
    train_loader: torch.utils.data.DataLoader,
    criterion: _Loss = None,
    criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor] = None,
    train_steps_fn: Callable[
        [
            torch.utils.data.DataLoader,
            torch.nn.Module,
            torch.optim.Optimizer,
            CompressionAlgorithmController,
            Optional[int],
        ],
        type(None),
    ] = None,
    validate_fn: Callable[[torch.nn.Module, torch.utils.data.DataLoader], Tuple[float, float]] = None,
    val_loader: torch.utils.data.DataLoader = None,
    autoq_eval_fn: Callable[[torch.nn.Module, torch.utils.data.DataLoader], float] = None,
    model_eval_fn: Callable[[torch.nn.Module, torch.utils.data.DataLoader], float] = None,
    distributed_callbacks: Tuple[Callable, Callable] = None,
    execution_parameters: ExecutionParameters = None,
    legr_train_optimizer: torch.optim.Optimizer = None,
    device: str = None,
) -> NNCFConfig:
    nncf_config.register_extra_structs(
        [
            QuantizationRangeInitArgs(data_loader=wrap_dataloader_for_init(train_loader), device=device),
            BNAdaptationInitArgs(data_loader=wrap_dataloader_for_init(train_loader), device=device),
        ]
    )
    if train_loader and train_steps_fn and val_loader and validate_fn:
        nncf_config.register_extra_structs(
            [
                LeGRInitArgs(
                    train_loader=train_loader,
                    train_fn=train_steps_fn,
                    val_loader=val_loader,
                    val_fn=validate_fn,
                    train_optimizer=legr_train_optimizer,
                    nncf_config=nncf_config,
                )
            ]
        )

    if criterion is not None:
        if not criterion_fn:
            criterion_fn = default_criterion_fn
        nncf_config.register_extra_structs(
            [
                QuantizationPrecisionInitArgs(
                    criterion_fn=criterion_fn, criterion=criterion, data_loader=train_loader, device=device
                )
            ]
        )

    if autoq_eval_fn is not None:
        if not val_loader:
            val_loader = train_loader
        nncf_config.register_extra_structs(
            [AutoQPrecisionInitArgs(data_loader=val_loader, eval_fn=autoq_eval_fn, nncf_config=nncf_config)]
        )

    if model_eval_fn is not None:
        nncf_config.register_extra_structs([ModelEvaluationArgs(eval_fn=model_eval_fn)])

    if distributed_callbacks is None:
        distributed_callbacks = (
            partial(default_distributed_wrapper, execution_parameters=execution_parameters),
            default_distributed_unwrapper,
        )
    else:
        nncf_logger.info("Utilizing user-provided distributed training wrappers.")

    nncf_config.register_extra_structs([DistributedCallbacksArgs(*distributed_callbacks)])
    return nncf_config

class PartialDataLoader:
    def __init__(self, regular_data_loader: DataLoader, iter_ratio=1.0):
        if iter_ratio < 0.0 or iter_ratio > 1.0:
            raise ValueError("iter_ratio must be within 0 to 1 range")
        self.data_loader = regular_data_loader
        self.batch_size = regular_data_loader.batch_size
        self._stop_id = math.ceil(len(self.data_loader) * iter_ratio)
        self._batch_id = 0

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self._batch_id = 0
        return self

    def __next__(self) -> Any:
        if self._batch_id < self._stop_id:
            loaded_item = next(self.data_loader_iter)
            self._batch_id += 1
            return loaded_item
        raise StopIteration

    def __len__(self) -> int:
        return self._stop_id

class DefaultInitializingDataLoader(PTInitializingDataLoader):
    def get_inputs(self, dataloader_output: Any) -> Tuple[Tuple, Dict]:
        return (dataloader_output[0],), {}

    def get_target(self, dataloader_output: Any):
        return dataloader_output[1]

