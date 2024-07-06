def assert_close(a: Tensor, b: Union[Tensor, List[Any]]) -> None:
    """Assert that ``a`` and ``b`` are element-wise equal within a tolerance."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b, device=device, dtype=a.dtype)

    torch.testing.assert_close(a, b)  # type: ignore[attr-defined]

def assert_equal(a: Tensor, b: Union[Tensor, List[Any]]) -> None:
    """Assert that ``a`` and ``b`` are element-wise equal."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b, device=device, dtype=a.dtype)

    torch.testing.assert_close(a, b, rtol=0, atol=0)  # type: ignore[attr-defined]

