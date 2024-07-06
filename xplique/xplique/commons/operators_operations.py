def check_operator(operator: Callable):
    """
    Check if the operator is valid g(f, x, y) -> tf.Tensor
    and raise an exception and return true if so.

    Parameters
    ----------
    operator
        Operator to check

    Returns
    -------
    is_valid
        True if the operator is valid, False otherwise.
    """
    # handle tf functions
    # pylint: disable=protected-access
    if hasattr(operator, '_python_function'):
        return check_operator(operator._python_function)

    # the operator must be callable
    if not hasattr(operator, '__call__'):
        raise_invalid_operator()

    # the operator should take at least three arguments
    args = inspect.getfullargspec(operator).args
    if len(args) < 3:
        raise_invalid_operator()

    return True

