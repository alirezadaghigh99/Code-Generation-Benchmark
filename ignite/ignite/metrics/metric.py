def sync_all_reduce(*attrs: Any) -> Callable:
    """Helper decorator for distributed configuration to collect instance attribute value
    across all participating processes and apply the specified reduction operation.

    See :doc:`metrics` on how to use it.

    Args:
        attrs: attribute names of decorated class

    .. versionchanged:: 0.4.5
        - Ability to handle different reduction operations (SUM, MAX, MIN, PRODUCT).
    """

    def wrapper(func: Callable) -> Callable:
        @wraps(func)
        def another_wrapper(self: Metric, *args: Any, **kwargs: Any) -> Callable:
            if not isinstance(self, Metric):
                raise RuntimeError(
                    "Decorator sync_all_reduce should be used on ignite.metric.Metric class methods only"
                )
            ws = idist.get_world_size()
            unreduced_attrs = {}
            if len(attrs) > 0 and ws > 1:
                for attr in attrs:
                    op_kwargs = {}
                    if ":" in attr:
                        attr, op = attr.split(":")
                        valid_ops = ["MIN", "MAX", "SUM", "PRODUCT"]
                        if op not in valid_ops:
                            raise ValueError(f"Reduction operation is not valid (expected : {valid_ops}, got: {op}")
                        op_kwargs["op"] = op
                    if attr not in self.__dict__:
                        raise ValueError(f"Metric {type(self)} has no attribute named `{attr}`.")
                    t = getattr(self, attr)
                    if not isinstance(t, (Number, torch.Tensor)):
                        raise TypeError(
                            "Attribute provided to sync_all_reduce should be a "
                            f"number or tensor but `{attr}` has type {type(t)}"
                        )
                    unreduced_attrs[attr] = t
                    # Here `clone` is necessary since `idist.all_reduce` modifies `t` inplace in the case
                    # `t` is a tensor and its `device` is same as that of the process.
                    # TODO: Remove this dual behavior of `all_reduce` to always either return a new tensor or
                    #       modify it in-place.
                    t_reduced = idist.all_reduce(cast(float, t) if isinstance(t, Number) else t.clone(), **op_kwargs)
                    setattr(self, attr, t_reduced)

            result = func(self, *args, **kwargs)

            for attr, value in unreduced_attrs.items():
                setattr(self, attr, value)
            return result

        return another_wrapper

    setattr(wrapper, "_decorated", True)
    return wrapper

def ITERATION_COMPLETED(self) -> CallableEventWithFilter:
        return self.__iteration_completed

