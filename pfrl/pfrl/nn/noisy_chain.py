def to_factorized_noisy(module, *args, **kwargs):
    """Add noisiness to components of given module

    Currently this fn. only supports torch.nn.Linear (with and without bias)
    """

    def func_to_factorized_noisy(module):
        if isinstance(module, nn.Linear):
            return FactorizedNoisyLinear(module, *args, **kwargs)
        else:
            return module

    _map_modules(func_to_factorized_noisy, module)