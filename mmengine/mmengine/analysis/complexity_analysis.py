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
    return giga_flops, flop_counter.unsupported_ops()

def parameter_count(model: nn.Module) -> typing.DefaultDict[str, int]:
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
    return count

def activation_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Optional[Dict[str, Handle]] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
    """Given a model and an input to the model, compute the total number of
    activations of the model.

    Adopted from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/activation_count.py

    Args:
        model (nn.Module): The model to compute activation counts.
        inputs (tuple): Inputs that are passed to `model` to count activations.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul. The key is operator name and the value
            is a function that takes (inputs, outputs) of the op.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
        activation (mega) for each operation and a Counter that records the
        number of unsupported operations.
    """
    if supported_ops is None:
        supported_ops = {}
    act_counter = ActivationAnalyzer(model,
                                     inputs).set_op_handle(**supported_ops)
    mega_acts = defaultdict(float)
    for op, act in act_counter.by_operator().items():
        mega_acts[op] = act / 1e6
    return mega_acts, act_counter.unsupported_ops()

class FlopAnalyzer(JitModelAnalysis):
    """Provides access to per-submodule model flop count obtained by tracing a
    model with pytorch's jit tracing functionality.

    By default, comes with standard flop counters for a few common operators.

    Note:
        - Flop is not a well-defined concept. We just produce our best
          estimate.
        - We count one fused multiply-add as one flop.

    Handles for additional operators may be added, or the default ones
    overwritten, using the ``.set_op_handle(name, func)`` method.
    See the method documentation for details.
    Flop counts can be obtained as:

    - ``.total(module_name="")``: total flop count for the module
    - ``.by_operator(module_name="")``: flop counts for the module, as a
      Counter over different operator types
    - ``.by_module()``: Counter of flop counts for all submodules
    - ``.by_module_and_operator()``: dictionary indexed by descendant of
      Counters over different operator types

    An operator is treated as within a module if it is executed inside the
    module's ``__call__`` method. Note that this does not include calls to
    other methods of the module or explicit calls to ``module.forward(...)``.

    Modified from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py

    Args:
        model (nn.Module): The model to analyze.
        inputs (Union[Tensor, Tuple[Tensor, ...]]): The input to the model.

    Examples:
        >>> import torch.nn as nn
        >>> import torch
        >>> class TestModel(nn.Module):
        ...    def __init__(self):
        ...        super().__init__()
        ...        self.fc = nn.Linear(in_features=1000, out_features=10)
        ...        self.conv = nn.Conv2d(
        ...            in_channels=3, out_channels=10, kernel_size=1
        ...        )
        ...        self.act = nn.ReLU()
        ...    def forward(self, x):
        ...        return self.fc(self.act(self.conv(x)).flatten(1))
        >>> model = TestModel()
        >>> inputs = (torch.randn((1,3,10,10)),)
        >>> flops = FlopAnalyzer(model, inputs)
        >>> flops.total()
        13000
        >>> flops.total("fc")
        10000
        >>> flops.by_operator()
        Counter({"addmm" : 10000, "conv" : 3000})
        >>> flops.by_module()
        Counter({"" : 13000, "fc" : 10000, "conv" : 3000, "act" : 0})
        >>> flops.by_module_and_operator()
        {"" : Counter({"addmm" : 10000, "conv" : 3000}),
        "fc" : Counter({"addmm" : 10000}),
        "conv" : Counter({"conv" : 3000}),
        "act" : Counter()
        }
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        super().__init__(model=model, inputs=inputs)
        self.set_op_handle(**_DEFAULT_SUPPORTED_FLOP_OPS)

    __init__.__doc__ = JitModelAnalysis.__init__.__doc__

class ActivationAnalyzer(JitModelAnalysis):
    """Provides access to per-submodule model activation count obtained by
    tracing a model with pytorch's jit tracing functionality.

    By default, comes with standard activation counters for convolutional and
    dot-product operators. Handles for additional operators may be added, or
    the default ones overwritten, using the ``.set_op_handle(name, func)``
    method. See the method documentation for details. Activation counts can be
    obtained as:

    - ``.total(module_name="")``: total activation count for a module
    - ``.by_operator(module_name="")``: activation counts for the module,
      as a Counter over different operator types
    - ``.by_module()``: Counter of activation counts for all submodules
    - ``.by_module_and_operator()``: dictionary indexed by descendant of
      Counters over different operator types

    An operator is treated as within a module if it is executed inside the
    module's ``__call__`` method. Note that this does not include calls to
    other methods of the module or explicit calls to ``module.forward(...)``.

    Modified from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/activation_count.py

    Args:
        model (nn.Module): The model to analyze.
        inputs (Union[Tensor, Tuple[Tensor, ...]]): The input to the model.

    Examples:
        >>> import torch.nn as nn
        >>> import torch
        >>> class TestModel(nn.Module):
        ...     def __init__(self):
        ...        super().__init__()
        ...        self.fc = nn.Linear(in_features=1000, out_features=10)
        ...        self.conv = nn.Conv2d(
        ...            in_channels=3, out_channels=10, kernel_size=1
        ...        )
        ...        self.act = nn.ReLU()
        ...    def forward(self, x):
        ...        return self.fc(self.act(self.conv(x)).flatten(1))
        >>> model = TestModel()
        >>> inputs = (torch.randn((1,3,10,10)),)
        >>> acts = ActivationAnalyzer(model, inputs)
        >>> acts.total()
        1010
        >>> acts.total("fc")
        10
        >>> acts.by_operator()
        Counter({"conv" : 1000, "addmm" : 10})
        >>> acts.by_module()
        Counter({"" : 1010, "fc" : 10, "conv" : 1000, "act" : 0})
        >>> acts.by_module_and_operator()
        {"" : Counter({"conv" : 1000, "addmm" : 10}),
        "fc" : Counter({"addmm" : 10}),
        "conv" : Counter({"conv" : 1000}),
        "act" : Counter()
        }
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        super().__init__(model=model, inputs=inputs)
        self.set_op_handle(**_DEFAULT_SUPPORTED_ACT_OPS)

    __init__.__doc__ = JitModelAnalysis.__init__.__doc__

