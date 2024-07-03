def flop_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Optional[Dict[str, Handle]] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
    """Given a model and an input to the model, compute the per-operator Gflops
    of the given model.

    Adopted from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py

    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul and einsum. The key is operator name and
            the value is a function that takes (inputs, outputs) of the op.
            We count one Multiply-Add as one FLOP.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
        gflops for each operation and a Counter that records the number of
        unsupported operations.
    """
    if supported_ops is None:
        supported_ops = {}
    flop_counter = FlopAnalyzer(model, inputs).set_op_handle(**supported_ops)
    giga_flops = defaultdict(float)
    for op, flop in flop_counter.by_operator().items():
        giga_flops[op] = flop / 1e9
    return giga_flops, flop_counter.unsupported_ops()def parameter_count(model: nn.Module) -> typing.DefaultDict[str, int]:
    """Count parameters of a model and its submodules.

    Adopted from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/parameter_count.py

    Args:
        model (nn.Module): the model to count parameters.

    Returns:
        dict[str, int]: the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
        number of parameters of the model.
    """
    count = defaultdict(int)  # type: typing.DefaultDict[str, int]
    for name, param in model.named_parameters():
        size = param.numel()
        name = name.split('.')
        for k in range(0, len(name) + 1):
            prefix = '.'.join(name[:k])
            count[prefix] += size
    return countdef flop_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Optional[Dict[str, Handle]] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
    """Given a model and an input to the model, compute the per-operator Gflops
    of the given model.

    Adopted from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py

    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul and einsum. The key is operator name and
            the value is a function that takes (inputs, outputs) of the op.
            We count one Multiply-Add as one FLOP.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
        gflops for each operation and a Counter that records the number of
        unsupported operations.
    """
    if supported_ops is None:
        supported_ops = {}
    flop_counter = FlopAnalyzer(model, inputs).set_op_handle(**supported_ops)
    giga_flops = defaultdict(float)
    for op, flop in flop_counter.by_operator().items():
        giga_flops[op] = flop / 1e9
    return giga_flops, flop_counter.unsupported_ops()