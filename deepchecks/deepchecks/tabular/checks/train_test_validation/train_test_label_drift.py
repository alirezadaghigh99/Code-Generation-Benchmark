class TrainTestLabelDrift(LabelDrift):
    """The TrainTestLabelDrift check is deprecated and will be removed in the 0.14 version.

    Please use the LabelDrift check instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The TrainTestLabelDrift check is deprecated and will be removed in the 0.14 version."
                      "Please use the LabelDrift check instead.", DeprecationWarning, stacklevel=2)
        LabelDrift.__init__(self, *args, **kwargs)

class TrainTestLabelDrift(LabelDrift):
    """The TrainTestLabelDrift check is deprecated and will be removed in the 0.14 version.

    Please use the LabelDrift check instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The TrainTestLabelDrift check is deprecated and will be removed in the 0.14 version."
                      "Please use the LabelDrift check instead.", DeprecationWarning, stacklevel=2)
        LabelDrift.__init__(self, *args, **kwargs)

