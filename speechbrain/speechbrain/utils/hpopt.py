class GenericHyperparameterOptimizationReporter(
    HyperparameterOptimizationReporter
):
    """
    A generic hyperparameter fit reporter that outputs the result as
    JSON to an arbitrary data stream, which may be read as a third-party
    tool

    Arguments
    ---------
    reference_date: datetime.datetime
        The date used to create trial id
    output: stream
        The stream to report the results to
    *args: tuple
        Arguments to be forwarded to parent class
    **kwargs: dict
        Arguments to be forwarded to parent class
    """

    def __init__(self, reference_date=None, output=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output or sys.stdout
        self.reference_date = reference_date
        self._trial_id = None

    def report_objective(self, result):
        """Reports the objective for hyperparameter optimization.

        Arguments
        ---------
        result: dict
            a dictionary with the run result.

        Example
        -------
        >>> reporter = GenericHyperparameterOptimizationReporter(
        ...     objective_key="error"
        ... )
        >>> result = {"error": 1.2, "train_loss": 7.2}
        >>> reporter.report_objective(result)
        {"error": 1.2, "train_loss": 7.2, "objective": 1.2}
        """
        json.dump(
            dict(result, objective=result[self.objective_key]), self.output
        )

    @property
    def trial_id(self):
        """The unique ID of this trial (used mainly for folder naming)

        Example
        -------
        >>> import datetime
        >>> reporter = GenericHyperparameterOptimizationReporter(
        ...     objective_key="error",
        ...     reference_date=datetime.datetime(2021, 1, 3)
        ... )
        >>> print(reporter.trial_id)
        20210103000000000000
        """
        if self._trial_id is None:
            date = self.reference_date or datetime.now()
            self._trial_id = date.strftime(FORMAT_TIMESTAMP)
        return self._trial_id

