class TrainTestFeatureDrift(FeatureDrift):
    """The TrainTestFeatureDrift check is deprecated and will be removed in the 0.14 version.

    .. deprecated:: 0.14.0
        `deepchecks.tabular.checks.TrainTestFeatureDrift is deprecated and will be removed in deepchecks 0.14 version.
        Use `deepchecks.tabular.checks.FeatureDrift` instead.

    Please use the FeatureDrift check instead
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The TrainTestFeatureDrift check is deprecated and will be removed in the 0.14 version."
                      " Please use the FeatureDrift check instead", DeprecationWarning, stacklevel=2)
        FeatureDrift.__init__(self, *args, **kwargs)

