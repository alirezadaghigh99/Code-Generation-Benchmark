class OptimizationContext:
    key: ClassVar[str] = "opt_ctx"

    dtype: Optional[torch.dtype] = None
    ops_name: str = ""

