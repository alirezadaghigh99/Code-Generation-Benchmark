def make_functional(model: nn.Module, disable_autograd_tracking: bool = False):
    warn_deprecated("make_functional", "torch.func.functional_call")
    return _nn_impl.make_functional(model, disable_autograd_tracking)

