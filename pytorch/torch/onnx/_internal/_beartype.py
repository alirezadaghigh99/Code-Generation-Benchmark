def _create_beartype_decorator(
    runtime_check_state: RuntimeTypeCheckState,
):
    # beartype needs to be imported outside of the function and aliased because
    # this module overwrites the name "beartype".

    if runtime_check_state == RuntimeTypeCheckState.DISABLED:
        return _no_op_decorator
    if _beartype_lib is None:
        # If the beartype library is not installed, return a no-op decorator
        return _no_op_decorator

    assert isinstance(_beartype_lib, ModuleType)

    if runtime_check_state == RuntimeTypeCheckState.ERRORS:
        # Enable runtime type checking which errors on any type hint violation.
        return _beartype_lib.beartype

    # Warnings only
    def beartype(func):
        """Warn on type hint violation."""

        if "return" in func.__annotations__:
            # Remove the return type from the func function's
            # annotations so that the beartype decorator does not complain
            # about the return type.
            return_type = func.__annotations__["return"]
            del func.__annotations__["return"]
            beartyped = _beartype_lib.beartype(func)
            # Restore the return type to the func function's annotations
            func.__annotations__["return"] = return_type
        else:
            beartyped = _beartype_lib.beartype(func)

        @functools.wraps(func)
        def _coerce_beartype_exceptions_to_warnings(*args, **kwargs):
            try:
                return beartyped(*args, **kwargs)
            except _roar.BeartypeCallHintParamViolation:
                # Fall back to the original function if the beartype hint is violated.
                warnings.warn(
                    traceback.format_exc(),
                    category=CallHintViolationWarning,
                    stacklevel=2,
                )

            return func(*args, **kwargs)  # noqa: B012

        return _coerce_beartype_exceptions_to_warnings

    return beartype