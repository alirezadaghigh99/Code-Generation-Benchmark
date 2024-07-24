class AOTConfig:
    """
    Configuration for AOTDispatcher
    """

    fw_compiler: Callable
    bw_compiler: Callable
    partition_fn: Callable
    decompositions: Dict[Callable, Callable]
    num_params_buffers: int
    aot_id: int
    keep_inference_input_mutations: bool
    is_export: bool = False
    no_tangents: bool = False
    dynamic_shapes: bool = False
    aot_autograd_arg_pos_to_source: Optional[List[Source]] = None
    inference_compiler: Optional[Callable] = None
    enable_log: bool = True
    # this is always false outside of export.
    pre_dispatch: bool = False

    # Key to use for AOTAutogradCache
    cache_key: Optional[str] = None

    def __post_init__(self):
        if self.pre_dispatch:
            assert self.is_export, "Can only have pre_dispatch IR for export."

