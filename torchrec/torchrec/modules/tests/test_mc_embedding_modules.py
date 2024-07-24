class Tracer(torch.fx.Tracer):
    _leaf_module_names: List[str]

    def __init__(self, leaf_module_names: Optional[List[str]] = None) -> None:
        super().__init__()
        self._leaf_module_names = leaf_module_names or []

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if (
            type(m).__name__ in self._leaf_module_names
            or module_qualified_name in self._leaf_module_names
        ):
            return True
        return super().is_leaf_module(m, module_qualified_name)

