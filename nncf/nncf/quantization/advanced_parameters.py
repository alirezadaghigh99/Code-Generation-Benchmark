class FP8QuantizationParameters:
    """
    Contains convert parameters for weights or activations.

    :param destination_type: Currently contains E4M3 or E5M2 for FP8 precision.
    :type destination_type: FP8Type
    """

    destination_type: Optional[FP8Type] = None

