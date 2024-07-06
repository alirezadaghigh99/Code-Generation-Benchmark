def has_torch_function(vt: "torch._dynamo.variables.base.VariableTracker") -> bool:
    from torch._dynamo.variables import LazyVariableTracker, UserDefinedObjectVariable
    from torch._dynamo.variables.torch_function import TensorWithTFOverrideVariable

    if isinstance(vt, TensorWithTFOverrideVariable):
        return True

    if isinstance(vt, LazyVariableTracker):
        LazyVariableTracker.realize(vt)

    return isinstance(vt, UserDefinedObjectVariable) and hasattr(
        vt.value, "__torch_function__"
    )

