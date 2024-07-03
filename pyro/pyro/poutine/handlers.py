def trace(
    fn: None = ...,
    graph_type: Optional[Literal["flat", "dense"]] = None,
    param_only: Optional[bool] = None,
) -> TraceMessenger: ...def block(
    fn: None = ...,
    hide_fn: Optional[Callable[["Message"], Optional[bool]]] = None,
    expose_fn: Optional[Callable[["Message"], Optional[bool]]] = None,
    hide_all: bool = True,
    expose_all: bool = False,
    hide: Optional[List[str]] = None,
    expose: Optional[List[str]] = None,
    hide_types: Optional[List[str]] = None,
    expose_types: Optional[List[str]] = None,
) -> BlockMessenger: ...def trace(
    fn: None = ...,
    graph_type: Optional[Literal["flat", "dense"]] = None,
    param_only: Optional[bool] = None,
) -> TraceMessenger: ...