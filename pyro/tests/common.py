def assert_equal(actual, expected, prec=1e-5, msg=""):
    if prec > 0.0:
        return assert_close(actual, expected, atol=prec, msg=msg)
    if not msg:
        msg = "{} vs {}".format(actual, expected)
    if isinstance(actual, numbers.Number) and isinstance(expected, numbers.Number):
        assert actual == expected, msg
    # Placing this as a second check allows for coercing of numeric types above;
    # this can be moved up to harden type checks.
    elif type(actual) != type(expected):
        raise AssertionError(
            "cannot compare {} and {}".format(type(actual), type(expected))
        )
    elif torch.is_tensor(actual) and torch.is_tensor(expected):
        assert actual.is_sparse == expected.is_sparse, msg
        if actual.is_sparse:
            x = _safe_coalesce(actual)
            y = _safe_coalesce(expected)
            assert_tensors_equal(x._indices(), y._indices(), msg=msg)
            assert_tensors_equal(x._values(), y._values(), msg=msg)
        else:
            assert_tensors_equal(actual, expected, msg=msg)
    elif type(actual) == np.ndarray and type(actual) == np.ndarray:
        assert (actual == expected).all(), msg
    elif isinstance(actual, dict):
        assert set(actual.keys()) == set(expected.keys())
        for key, x_val in actual.items():
            assert_equal(
                x_val,
                expected[key],
                prec=0.0,
                msg="At key{}: {} vs {}".format(key, x_val, expected[key]),
            )
    elif isinstance(actual, str):
        assert actual == expected, msg
    elif is_iterable(actual) and is_iterable(expected):
        assert len(actual) == len(expected), msg
        for xi, yi in zip(actual, expected):
            assert_equal(xi, yi, prec=0.0, msg="{} vs {}".format(xi, yi))
    else:
        assert actual == expected, msgdef assert_close(actual, expected, atol=1e-7, rtol=0, msg=""):
    if not msg:
        msg = "{} vs {}".format(actual, expected)
    if isinstance(actual, numbers.Number) and isinstance(expected, numbers.Number):
        assert actual == approx(expected, abs=atol, rel=rtol), msg
    # Placing this as a second check allows for coercing of numeric types above;
    # this can be moved up to harden type checks.
    elif type(actual) != type(expected):
        raise AssertionError(
            "cannot compare {} and {}".format(type(actual), type(expected))
        )
    elif torch.is_tensor(actual) and torch.is_tensor(expected):
        prec = atol + rtol * abs(expected) if rtol > 0 else atol
        assert actual.is_sparse == expected.is_sparse, msg
        if actual.is_sparse:
            x = _safe_coalesce(actual)
            y = _safe_coalesce(expected)
            assert_tensors_equal(x._indices(), y._indices(), prec, msg)
            assert_tensors_equal(x._values(), y._values(), prec, msg)
        else:
            assert_tensors_equal(actual, expected, prec, msg)
    elif type(actual) == np.ndarray and type(expected) == np.ndarray:
        assert_allclose(
            actual, expected, atol=atol, rtol=rtol, equal_nan=True, err_msg=msg
        )
    elif isinstance(actual, numbers.Number) and isinstance(y, numbers.Number):
        assert actual == approx(expected, abs=atol, rel=rtol), msg
    elif isinstance(actual, dict):
        assert set(actual.keys()) == set(expected.keys())
        for key, x_val in actual.items():
            assert_close(
                x_val,
                expected[key],
                atol=atol,
                rtol=rtol,
                msg="At key {}: {} vs {}".format(repr(key), x_val, expected[key]),
            )
    elif isinstance(actual, str):
        assert actual == expected, msg
    elif is_iterable(actual) and is_iterable(expected):
        assert len(actual) == len(expected), msg
        for xi, yi in zip(actual, expected):
            assert_close(xi, yi, atol=atol, rtol=rtol, msg="{} vs {}".format(xi, yi))
    else:
        assert actual == expected, msgdef assert_equal(actual, expected, prec=1e-5, msg=""):
    if prec > 0.0:
        return assert_close(actual, expected, atol=prec, msg=msg)
    if not msg:
        msg = "{} vs {}".format(actual, expected)
    if isinstance(actual, numbers.Number) and isinstance(expected, numbers.Number):
        assert actual == expected, msg
    # Placing this as a second check allows for coercing of numeric types above;
    # this can be moved up to harden type checks.
    elif type(actual) != type(expected):
        raise AssertionError(
            "cannot compare {} and {}".format(type(actual), type(expected))
        )
    elif torch.is_tensor(actual) and torch.is_tensor(expected):
        assert actual.is_sparse == expected.is_sparse, msg
        if actual.is_sparse:
            x = _safe_coalesce(actual)
            y = _safe_coalesce(expected)
            assert_tensors_equal(x._indices(), y._indices(), msg=msg)
            assert_tensors_equal(x._values(), y._values(), msg=msg)
        else:
            assert_tensors_equal(actual, expected, msg=msg)
    elif type(actual) == np.ndarray and type(actual) == np.ndarray:
        assert (actual == expected).all(), msg
    elif isinstance(actual, dict):
        assert set(actual.keys()) == set(expected.keys())
        for key, x_val in actual.items():
            assert_equal(
                x_val,
                expected[key],
                prec=0.0,
                msg="At key{}: {} vs {}".format(key, x_val, expected[key]),
            )
    elif isinstance(actual, str):
        assert actual == expected, msg
    elif is_iterable(actual) and is_iterable(expected):
        assert len(actual) == len(expected), msg
        for xi, yi in zip(actual, expected):
            assert_equal(xi, yi, prec=0.0, msg="{} vs {}".format(xi, yi))
    else:
        assert actual == expected, msgdef assert_close(actual, expected, atol=1e-7, rtol=0, msg=""):
    if not msg:
        msg = "{} vs {}".format(actual, expected)
    if isinstance(actual, numbers.Number) and isinstance(expected, numbers.Number):
        assert actual == approx(expected, abs=atol, rel=rtol), msg
    # Placing this as a second check allows for coercing of numeric types above;
    # this can be moved up to harden type checks.
    elif type(actual) != type(expected):
        raise AssertionError(
            "cannot compare {} and {}".format(type(actual), type(expected))
        )
    elif torch.is_tensor(actual) and torch.is_tensor(expected):
        prec = atol + rtol * abs(expected) if rtol > 0 else atol
        assert actual.is_sparse == expected.is_sparse, msg
        if actual.is_sparse:
            x = _safe_coalesce(actual)
            y = _safe_coalesce(expected)
            assert_tensors_equal(x._indices(), y._indices(), prec, msg)
            assert_tensors_equal(x._values(), y._values(), prec, msg)
        else:
            assert_tensors_equal(actual, expected, prec, msg)
    elif type(actual) == np.ndarray and type(expected) == np.ndarray:
        assert_allclose(
            actual, expected, atol=atol, rtol=rtol, equal_nan=True, err_msg=msg
        )
    elif isinstance(actual, numbers.Number) and isinstance(y, numbers.Number):
        assert actual == approx(expected, abs=atol, rel=rtol), msg
    elif isinstance(actual, dict):
        assert set(actual.keys()) == set(expected.keys())
        for key, x_val in actual.items():
            assert_close(
                x_val,
                expected[key],
                atol=atol,
                rtol=rtol,
                msg="At key {}: {} vs {}".format(repr(key), x_val, expected[key]),
            )
    elif isinstance(actual, str):
        assert actual == expected, msg
    elif is_iterable(actual) and is_iterable(expected):
        assert len(actual) == len(expected), msg
        for xi, yi in zip(actual, expected):
            assert_close(xi, yi, atol=atol, rtol=rtol, msg="{} vs {}".format(xi, yi))
    else:
        assert actual == expected, msg