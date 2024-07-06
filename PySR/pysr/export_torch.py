def sympy2torch(expression, symbols_in, selection=None, extra_torch_mappings=None):
    """Returns a module for a given sympy expression with trainable parameters;

    This function will assume the input to the module is a matrix X, where
        each column corresponds to each symbol you pass in `symbols_in`.
    """
    global SingleSymPyModule

    _initialize_torch()

    return SingleSymPyModule(
        expression, symbols_in, selection=selection, extra_funcs=extra_torch_mappings
    )

