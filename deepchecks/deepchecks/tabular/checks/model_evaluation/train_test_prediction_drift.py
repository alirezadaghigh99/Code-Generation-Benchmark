class TrainTestPredictionDrift(PredictionDrift):
    """The TrainTestPredictionDrift check is deprecated and will be removed in the 0.14 version.

    Please use the PredictionDrift check instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The TrainTestPredictionDrift check is deprecated and will be removed in the 0.14 version. "
                      "Please use the PredictionDrift check instead.", DeprecationWarning, stacklevel=2)
        PredictionDrift.__init__(self, *args, **kwargs)

class TrainTestPredictionDrift(PredictionDrift):
    """The TrainTestPredictionDrift check is deprecated and will be removed in the 0.14 version.

    Please use the PredictionDrift check instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The TrainTestPredictionDrift check is deprecated and will be removed in the 0.14 version. "
                      "Please use the PredictionDrift check instead.", DeprecationWarning, stacklevel=2)
        PredictionDrift.__init__(self, *args, **kwargs)

