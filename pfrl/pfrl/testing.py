def torch_assert_allclose(actual, desired, *args, **kwargs):
    """Assert two objects are equal up to desired tolerance.

    This function can be used as a replacement of
    `numpy.testing.assert_allclose` except that lists, tuples, and
    `torch.Tensor`s are converted to `numpy.ndarray`s automatically before
    comparison.
    """
    actual = _as_numpy_recursive(actual)
    desired = _as_numpy_recursive(desired)
    np.testing.assert_allclose(actual, desired, *args, **kwargs)

