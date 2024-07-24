class LayoutDescriptor:
    """
    Container to store convolutional and linear layers layout information.
    """

    conv_weight_out_channels_dim: int
    conv_weight_in_channels_dim: int
    bias_channels_dim: int

