class FSDPStrategy(Strategy):
    """Dataclass representing the `FullyShardedDataParallel <https://pytorch.org/docs/stable/fsdp.html>`_ strategy"""

    process_group: Optional[ProcessGroup] = None
    sharding_strategy: Optional[Union[str, _ShardingStrategy]] = None
    cpu_offload: Optional[CPUOffload] = None
    auto_wrap_policy: Optional[Callable[[torch.nn.Module, bool, int], bool]] = None
    backward_prefetch: Optional[Union[str, _BackwardPrefetch]] = (
        _BackwardPrefetch.BACKWARD_PRE
    )
    mixed_precision: Optional[Union[_MixedPrecision, MixedPrecision]] = None
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None
    param_init_fn: Optional[Callable[[torch.nn.Module], None]] = None
    sync_module_states: bool = False
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = False

    # FSDP set_state_dict_type params: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type
    # for setting type of state dict for checkpointing
    state_dict_type: Optional[Union[str, _StateDictType]] = None
    state_dict_config: Optional[StateDictConfig] = None
    optim_state_dict_config: Optional[OptimStateDictConfig] = None

    def __post_init__(self) -> None:
        if isinstance(self.sharding_strategy, str):
            self.sharding_strategy = ShardingStrategy.to_native_sharding_strategy(
                self.sharding_strategy
            )

        if isinstance(self.backward_prefetch, str):
            self.backward_prefetch = BackwardPrefetch.to_native_backward_prefetch(
                self.backward_prefetch
            )

        if isinstance(self.state_dict_type, str):
            self.state_dict_type = StateDictType.to_native_state_dict_type(
                self.state_dict_type
            )

        if isinstance(self.mixed_precision, MixedPrecision):
            self.mixed_precision = self.mixed_precision.to_native_mixed_precision()

