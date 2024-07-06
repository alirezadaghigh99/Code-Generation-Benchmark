def register(self, *types: Types, **kwargs: Any) -> Callable[[_C], _C]:
        # Override to add type hints...
        result: Callable[[_C], _C] = super().register(*types, **kwargs)
        return result

