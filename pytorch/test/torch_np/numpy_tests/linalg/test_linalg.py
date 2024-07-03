def assert_almost_equal(a, b, single_decimal=6, double_decimal=12, **kw):
    if asarray(a).dtype.type in (single, csingle):
        decimal = single_decimal
    else:
        decimal = double_decimal
    old_assert_almost_equal(a, b, decimal=decimal, **kw)