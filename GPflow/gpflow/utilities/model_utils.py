def assert_params_false(
    called_method: Callable[..., Any],
    **kwargs: bool,
) -> None:
    """
    Asserts that parameters are ``False``.

    :param called_method: The method or function that is calling this. Used for nice error messages.
    :param kwargs: Parameters that must be ``False``.
    :raises NotImplementedError: If any ``kwargs`` are ``True``.
    """
    errors_str = ", ".join(f"{param}={value}" for param, value in kwargs.items() if value)
    if errors_str:
        raise NotImplementedError(
            f"{called_method.__qualname__} does not currently support: {errors_str}"
        )

