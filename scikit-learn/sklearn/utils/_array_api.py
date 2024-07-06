def get_namespace(*arrays, remove_none=True, remove_types=(str,), xp=None):
    """Get namespace of arrays.

    Introspect `arrays` arguments and return their common Array API compatible
    namespace object, if any.

    See: https://numpy.org/neps/nep-0047-array-api-standard.html

    If `arrays` are regular numpy arrays, an instance of the `_NumPyAPIWrapper`
    compatibility wrapper is returned instead.

    Namespace support is not enabled by default. To enabled it call:

      sklearn.set_config(array_api_dispatch=True)

    or:

      with sklearn.config_context(array_api_dispatch=True):
          # your code here

    Otherwise an instance of the `_NumPyAPIWrapper` compatibility wrapper is
    always returned irrespective of the fact that arrays implement the
    `__array_namespace__` protocol or not.

    Parameters
    ----------
    *arrays : array objects
        Array objects.

    remove_none : bool, default=True
        Whether to ignore None objects passed in arrays.

    remove_types : tuple or list, default=(str,)
        Types to ignore in the arrays.

    xp : module, default=None
        Precomputed array namespace module. When passed, typically from a caller
        that has already performed inspection of its own inputs, skips array
        namespace inspection.

    Returns
    -------
    namespace : module
        Namespace shared by array objects. If any of the `arrays` are not arrays,
        the namespace defaults to NumPy.

    is_array_api_compliant : bool
        True if the arrays are containers that implement the Array API spec.
        Always False when array_api_dispatch=False.
    """
    array_api_dispatch = get_config()["array_api_dispatch"]
    if not array_api_dispatch:
        if xp is not None:
            return xp, False
        else:
            return _NUMPY_API_WRAPPER_INSTANCE, False

    if xp is not None:
        return xp, True

    arrays = _remove_non_arrays(
        *arrays, remove_none=remove_none, remove_types=remove_types
    )

    _check_array_api_dispatch(array_api_dispatch)

    # array-api-compat is a required dependency of scikit-learn only when
    # configuring `array_api_dispatch=True`. Its import should therefore be
    # protected by _check_array_api_dispatch to display an informative error
    # message in case it is missing.
    import array_api_compat

    namespace, is_array_api_compliant = array_api_compat.get_namespace(*arrays), True

    # These namespaces need additional wrapping to smooth out small differences
    # between implementations
    if namespace.__name__ in {"cupy.array_api"}:
        namespace = _ArrayAPIWrapper(namespace)

    if namespace.__name__ == "array_api_strict" and hasattr(
        namespace, "set_array_api_strict_flags"
    ):
        namespace.set_array_api_strict_flags(api_version="2023.12")

    return namespace, is_array_api_compliant

def yield_namespace_device_dtype_combinations(include_numpy_namespaces=True):
    """Yield supported namespace, device, dtype tuples for testing.

    Use this to test that an estimator works with all combinations.

    Parameters
    ----------
    include_numpy_namespaces : bool, default=True
        If True, also yield numpy namespaces.

    Returns
    -------
    array_namespace : str
        The name of the Array API namespace.

    device : str
        The name of the device on which to allocate the arrays. Can be None to
        indicate that the default value should be used.

    dtype_name : str
        The name of the data type to use for arrays. Can be None to indicate
        that the default value should be used.
    """
    for array_namespace in yield_namespaces(
        include_numpy_namespaces=include_numpy_namespaces
    ):
        if array_namespace == "torch":
            for device, dtype in itertools.product(
                ("cpu", "cuda"), ("float64", "float32")
            ):
                yield array_namespace, device, dtype
            yield array_namespace, "mps", "float32"
        else:
            yield array_namespace, None, None

