class TrainPipelineContext:
    """
    Context information for a `TrainPipelineSparseDist` instance.

    Attributes:
        input_dist_splits_requests (Dict[str, Awaitable[Any]]): Stores input dist
            requests in the splits awaitable stage, which occurs after starting the
            input dist.
        input_dist_tensors_requests (Dict[str, Awaitable[Any]]): Stores input dist
            requests in the tensors awaitable stage, which occurs after calling `wait()`
            on the splits awaitable.
        module_contexts (Dict[str, Multistreamable]): Stores module contexts from the
            input dist for the current batch.
        module_contexts_next_batch (Dict[str, Multistreamable]): Stores module contexts
            from the input dist for the next batch. (only for version 0)
        fused_splits_awaitables (List[Tuple[List[str], FusedKJTListSplitsAwaitable]]):
            List of fused splits input dist awaitable and the corresponding module names
            of each awaitable.
        event: Optional[torch.cuda.Event]: Event to record the completion of this stage
        index: Optional[int]: Index of the current batch.
        version: int = 0; support for backward compatiblity
    """

    # pyre-ignore [4]
    input_dist_splits_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    # pyre-ignore [4]
    input_dist_tensors_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    module_contexts: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_next_batch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )  # deprecated: to support legacy code
    fused_splits_awaitables: List[Tuple[List[str], FusedKJTListSplitsAwaitable]] = (
        field(default_factory=list)
    )
    events: List[torch.Event] = field(default_factory=list)
    preproc_fwd_results: Dict[str, Any] = field(default_factory=dict)
    index: Optional[int] = None
    version: int = (
        0  # 1 is current version, 0 is deprecated but supported for backward compatibility
    )

