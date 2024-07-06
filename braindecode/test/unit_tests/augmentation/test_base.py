def common_transform_assertions(
        input_batch,
        output_batch,
        expected_X=None,
        diff_param=None,
):
    """Assert whether shapes and devices are conserved. Also, (optional)
    checks whether the expected features matrix is produced.

    Parameters
    ----------
    input_batch : tuple
        The batch given to the transform containing a tensor X, of shape
        (batch_sizze, n_channels, sequence_len), and a tensor  y of shape
        (batch_size).
    output_batch : tuple
        The batch output by the transform. Should have two elements: the
        transformed X and y.
    expected_X : torch.Tensor, optional
        The expected first element of output_batch, which will be compared to
        it. By default None.
    diff_param : torch.Tensor | None, optional
        Parameter which should have grads.
    """
    X, y = input_batch
    tr_X, tr_y = output_batch
    assert tr_X.shape == X.shape
    assert tr_X.shape[0] == tr_y.shape[0]
    assert torch.equal(tr_y, y)
    assert X.device == tr_X.device
    if expected_X is not None:
        assert torch.equal(tr_X, expected_X)
    if diff_param is not None:
        loss = (tr_X - X).sum()
        loss.backward()
        assert diff_param.grad is not None

