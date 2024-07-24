class FakeQuantizeParameters:
    """
    Class handles FakeQuantize layer attributes.

    :param input_low: Tensor with minimum limit for input value.
    :param input_high: Tensor with maximum limit for input value.
    :param output_low: Tensor with minimum quantized value.
    :param output_high: Tensor with maximum quantized value.
    :param levels: Number of quantization levels.
    """

    input_low: Tensor
    input_high: Tensor
    output_low: Tensor
    output_high: Tensor
    levels: int

class FakeConvertParameters:
    """
    Class handles FakeConvert layer attributes.

    :param scale: Tensor with the scale for input value.
    :param shift: Tensor with the shift for input value.
    :param destination_type: Destination type.
    """

    scale: Tensor
    shift: Tensor
    destination_type: FP8Type

