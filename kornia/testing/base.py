def assert_close(
    actual: Tensor, expected: Tensor, *, rtol: Optional[float] = None, atol: Optional[float] = None, **kwargs: Any
) -> None:
    if rtol is None and atol is None:
        # `torch.testing.assert_close` used different default tolerances than `torch.testing.assert_allclose`.
        # TODO: remove this special handling as soon as https://github.com/kornia/kornia/issues/1134 is resolved
        #  Basically, this whole wrapper function can be removed and `torch.testing.assert_close` can be used
        #  directly.
        rtol, atol = _default_tolerances(actual, expected)

    return _assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        # this is the default value for torch>=1.10, but not for torch==1.9
        # TODO: remove this if kornia relies on torch>=1.10
        check_stride=False,
        equal_nan=False,
        **kwargs,
    )