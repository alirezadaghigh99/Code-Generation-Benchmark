class dual_level(_DecoratorContextManager):
    r"""Context-manager for forward AD, where all forward AD computation must occur within the ``dual_level`` context.

    .. Note::

        The ``dual_level`` context appropriately enters and exit the dual level to
        controls the current forward AD level, which is used by default by the other
        functions in this API.

        We currently don't plan to support nested ``dual_level`` contexts, however, so
        only a single forward AD level is supported. To compute higher-order
        forward grads, one can use :func:`torch.func.jvp`.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> x = torch.tensor([1])
        >>> x_t = torch.tensor([1])
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     # Do computations with inp
        ...     out = your_fn(inp)
        ...     _, grad = unpack_dual(out)
        >>> grad is None
        False
        >>> # After exiting the level, the grad is deleted
        >>> _, grad_after = unpack_dual(out)
        >>> grad is None
        True

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """

    def __enter__(self):
        return enter_dual_level()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        exit_dual_level()

class _set_fwd_grad_enabled(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None:
        self.prev = _is_fwd_grad_enabled()
        torch._C._set_fwd_grad_enabled(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_fwd_grad_enabled(self.prev)

