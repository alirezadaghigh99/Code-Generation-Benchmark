def has_arg(func, name, accept_all=False):
    """Checks if a callable accepts a given keyword argument.

    For Python 2, checks if there is an argument with the given name.
    For Python 3, checks if there is an argument with the given name, and also whether this
    argument can be called with a keyword (i.e. if it is not a positional-only argument).

    Parameters
    ----------
    func: object
        Callable to inspect.
    name: str
        Check if `func` can be called with `name` as a keyword argument.
    accept_all: bool, optional
        What to return if there is no parameter called `name` but the function accepts a
        `**kwargs` argument. Default: ``False``

    Returns
    -------
    bool
        Whether `func` accepts a `name` keyword argument.
    """
    signature = inspect.signature(func)
    parameter = signature.parameters.get(name)
    if parameter is None:
        if accept_all:
            for param in signature.parameters.values():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    return True
        return False
    return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY))