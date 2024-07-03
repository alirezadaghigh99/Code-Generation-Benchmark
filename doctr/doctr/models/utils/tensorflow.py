def conv_sequence(
    out_channels: int,
    activation: Optional[Union[str, Callable]] = None,
    bn: bool = False,
    padding: str = "same",
    kernel_initializer: str = "he_normal",
    **kwargs: Any,
) -> List[layers.Layer]:
    """Builds a convolutional-based layer sequence

    >>> from tensorflow.keras import Sequential
    >>> from doctr.models import conv_sequence
    >>> module = Sequential(conv_sequence(32, 'relu', True, kernel_size=3, input_shape=[224, 224, 3]))

    Args:
    ----
        out_channels: number of output channels
        activation: activation to be used (default: no activation)
        bn: should a batch normalization layer be added
        padding: padding scheme
        kernel_initializer: kernel initializer
        **kwargs: additional arguments to be passed to the convolutional layer

    Returns:
    -------
        list of layers
    """
    # No bias before Batch norm
    kwargs["use_bias"] = kwargs.get("use_bias", not bn)
    # Add activation directly to the conv if there is no BN
    kwargs["activation"] = activation if not bn else None
    conv_seq = [layers.Conv2D(out_channels, padding=padding, kernel_initializer=kernel_initializer, **kwargs)]

    if bn:
        conv_seq.append(layers.BatchNormalization())

    if (isinstance(activation, str) or callable(activation)) and bn:
        # Activation function can either be a string or a function ('relu' or tf.nn.relu)
        conv_seq.append(layers.Activation(activation))

    return conv_seq