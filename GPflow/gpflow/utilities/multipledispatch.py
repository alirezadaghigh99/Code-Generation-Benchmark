def register(self, *types: Types, **kwargs: Any) -> Callable[[_C], _C]:
        # Override to add type hints...
        result: Callable[[_C], _C] = super().register(*types, **kwargs)
        return result

class Dispatcher(GeneratorDispatcher):
    """
    multipledispatch.Dispatcher uses a generator to yield the
    desired function implementation, which is problematic as TensorFlow's
    autograph is not able to compile code that passes through generators.

    This class overwrites the problematic method in the original
    Dispatcher and solely makes use of simple for-loops, which are
    compilable by AutoGraph.
    """

    def register(self, *types: Types, **kwargs: Any) -> Callable[[_C], _C]:
        # Override to add type hints...
        result: Callable[[_C], _C] = super().register(*types, **kwargs)
        return result

    def dispatch(self, *types: Types) -> Optional[AnyCallable]:
        """
        Returns matching function for `types`; if not existing returns None.
        """
        if types in self.funcs:
            result: AnyCallable = self.funcs[types]
            return result

        return self.get_first_occurrence(*types)

    def dispatch_or_raise(self, *types: Types) -> AnyCallable:
        """
        Returns matching function for `types`; if not existing raises an error.
        """
        f = self.dispatch(*types)
        if f is None:
            raise NotImplementedError(
                f"Could not find signature for {self.name}: <{str_signature(types)}>"
            )
        return f

    def get_first_occurrence(self, *types: Types) -> Optional[AnyCallable]:
        """
        Returns the first occurrence of a matching function

        Based on `multipledispatch.Dispatcher.dispatch_iter`, which
        returns an iterator of matching functions. This method uses
        the same logic to select functions, but simply returns the first
        element of the iterator. If no matching functions are found,
        `None` is returned.
        """
        n = len(types)
        for signature in self.ordering:
            if len(signature) == n and all(map(issubclass, types, signature)):  # type: ignore[arg-type]
                result: AnyCallable = self.funcs[signature]
                return result
            elif len(signature) and isvariadic(signature[-1]):
                if variadic_signature_matches(types, signature):
                    result = self.funcs[signature]
                    return result
        return None

