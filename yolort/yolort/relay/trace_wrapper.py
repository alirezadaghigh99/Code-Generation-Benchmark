def get_trace_module(
    model_func: Callable[..., nn.Module],
    input_shape: Tuple[int, int] = (416, 416),
):
    """
    Get the tracing of a given model function.

    Example:

        >>> from yolort.models import yolov5s
        >>> from yolort.relaying.trace_wrapper import get_trace_module
        >>>
        >>> model = yolov5s(pretrained=True)
        >>> tracing_module = get_trace_module(model)
        >>> print(tracing_module.code)
        def forward(self,
            x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
          _0, _1, _2, = (self.model).forward(x, )
          return (_0, _1, _2)

    Args:
        model_func (Callable): The model function to be traced.
        input_shape (Tuple[int, int]): Shape size of the input image.
    """
    model = TraceWrapper(model_func)
    model.eval()

    dummy_input = torch.rand(1, 3, *input_shape)
    trace_module = torch.jit.trace(model, dummy_input)
    trace_module.eval()

    return trace_module