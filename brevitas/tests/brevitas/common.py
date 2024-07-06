def assert_allclose(generated, reference):
    assert torch.allclose(generated, reference, RTOL, ATOL)

