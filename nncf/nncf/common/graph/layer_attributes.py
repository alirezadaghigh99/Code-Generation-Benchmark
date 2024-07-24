class ConvolutionLayerAttributes(WeightedLayerAttributes):
    def __init__(
        self,
        weight_requires_grad: bool,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        dilations: Tuple[int, ...],
        groups: int,
        transpose: bool,
        padding_values: Union[Tuple[int, ...], int],
        with_bias: bool = False,
        output_padding_values: Optional[Union[Tuple[int, ...], int]] = None,
    ):
        """

        :param weight_requires_grad: Is True if gradients need to be computed for the corresponding Tensor,
        False otherwise.
        :param in_channels: Number of input channels in the layer's input.
        :param out_channels: Number of channels produced by the layer.
        :param kernel_size: Size of the convolving kernel.
        :param stride: Stride of the convolution.
        :param groups: Number of blocked connections from input channels to output channels.
        :param transpose: If set to `True`, the layer is an ordinary convolution, otherwise - transpose one.
        :param padding_values: Defines the amount of padding applied to the layer's input.
        :param with_bias: Operation include bias.
        :param output_padding_values: Defines the amount of output padding applied to the layer's output, for transpose.
        """
        super().__init__(weight_requires_grad=weight_requires_grad, with_bias=with_bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilations = dilations
        self.groups = groups
        self.transpose = transpose
        self.padding_values = padding_values
        self.output_padding_values = output_padding_values

    def get_weight_shape(self) -> List[int]:
        if not self.transpose:
            return [self.out_channels, self.in_channels // self.groups, *self.kernel_size]
        return [self.in_channels, self.out_channels // self.groups, *self.kernel_size]

    def get_target_dim_for_compression(self) -> int:
        # Always quantize per each "out" channel
        if self.transpose:
            return 1
        return 0

class MultipleInputLayerAttributes(BaseLayerAttributes):
    def __init__(self, axis: int, num_inputs: Optional[int] = None):
        """

        :param axis: the dimension over which the inputs are combined (e.g. concatenated).
        :param num_inputs: Number of inputs.
        """
        self.axis = axis
        self.num_inputs = num_inputs

class ReshapeLayerAttributes(BaseLayerAttributes):
    """
    :param input_shape: number of elements of each of the axes of a input tensor.
    :param output_shape: number of elements of each of the axes of a output tensor.
    """

    input_shape: List[int]
    output_shape: List[int]

