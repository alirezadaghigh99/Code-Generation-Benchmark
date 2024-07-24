class Metric(object):

    def __init__(self, name, lower_is_better=True):
        """
        Represents any metric.

        Parameters
        ----------
        name : str
            A name for the metric.
        lower_is_better : bool
            Indicates if the metric is better for lower values if True,
            otherwise it is assumed that higher values are better.
        """
        self.name = name
        self.lower_is_better = lower_is_better

