def _pad_shift_array(x, stride=1):
    """Zero-pad and shift rows of a 3D array.

    E.g., used to align predictions of corresponding windows in
    sequence-to-sequence models.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_rows, n_classes, n_windows).
    stride : int
        Number of non-overlapping elements between two consecutive sequences.

    Returns
    -------
    np.ndarray :
        Array of shape (n_rows, n_classes, (n_rows - 1) * stride + n_windows)
        where each row is obtained by zero-padding the corresponding row in
        ``x`` before and after in the last dimension.
    """
    if x.ndim != 3:
        raise NotImplementedError(
            "x must be of shape (n_rows, n_classes, n_windows), got " f"{x.shape}"
        )
    x_padded = np.pad(x, ((0, 0), (0, 0), (0, (x.shape[0] - 1) * stride)))
    orig_strides = x_padded.strides
    new_strides = (
        orig_strides[0] - stride * orig_strides[2],
        orig_strides[1],
        orig_strides[2],
    )
    return np.lib.stride_tricks.as_strided(x_padded, strides=new_strides)

def get_output_shape(model, in_chans, input_window_samples):
    """Returns shape of neural network output for batch size equal 1.

    Returns
    -------
    output_shape: tuple
        shape of the network output for `batch_size==1` (1, ...)
    """
    with torch.no_grad():
        dummy_input = torch.ones(
            1,
            in_chans,
            input_window_samples,
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )
        output_shape = model(dummy_input).shape
    return output_shape

