def check_equal(
        cls,
        test: Union[TensorType, List[TensorType]],
        reference: Union[TensorType, List[TensorType]],
        rtol: float = 1e-1,
        atol=0,
    ):
        cls._check_assertion(test, reference, lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol))

