def compute_fans(shape, data_format='channels_last'):
    """Computes the number of input and output units for a weight shape.

    Ported directly from Keras as the location moves between keras and tensorflow-keras

    Parameters
    ----------
    shape: tuple
        shape tuple of integers
    data_format: str
        Image data format to use for convolution kernels. Note that all kernels in Keras are
        standardized on the `"channels_last"` ordering (even when inputs are set to
        `"channels_first"`).

    Returns
    -------
    tuple
            A tuple of scalars, `(fan_in, fan_out)`.

    Raises
    ------
    ValueError
        In case of invalid `data_format` argument.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # Theano kernel shape: (depth, input_depth, ...)
        # Tensorflow kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out