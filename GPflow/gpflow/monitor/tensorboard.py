class ModelToTensorBoard(ToTensorBoard):
    """
    Monitoring task that creates a sensible TensorBoard for a model.

    Monitors all the model's parameters for which their name matches with `keywords_to_monitor`.
    By default, "kernel" and "likelihood" are elements of `keywords_to_monitor`.

    Example::

        keyword = "kernel", parameter = "kernel.lengthscale" => match
        keyword = "variational", parameter = "kernel.lengthscale" => no match
    """

    def __init__(
        self,
        log_dir: str,
        model: BayesianModel,
        *,
        max_size: int = 3,
        keywords_to_monitor: List[str] = ["kernel", "likelihood"],
        left_strip_character: str = ".",
    ) -> None:
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be a nested: for example, './logs/my_run/'.
        :param model: model to be monitord.
        :param max_size: maximum size of arrays (incl.) to store each
            element of the array independently as a scalar in the TensorBoard.
            Setting max_size to -1 will write all values. Use with care.
        :param keywords_to_monitor: specifies keywords to be monitored.
            If the parameter's name includes any of the keywords specified it
            will be monitored. By default, parameters that match the `kernel` or
            `likelihood` keyword are monitored.
            Adding a "*" to the list will match with all parameters,
            i.e. no parameters or variables will be filtered out.
        :param left_strip_character: certain frameworks prepend their variables with
            a character. GPflow adds a '.' and Keras add a '_', for example.
            When a `left_strip_character` is specified it will be stripped from the
            parameter's name. By default the '.' is left stripped, for example:
            ".likelihood.variance" becomes "likelihood.variance".
        """
        super().__init__(log_dir)
        self.model = model
        self.max_size = max_size
        self.keywords_to_monitor = keywords_to_monitor
        self.summarize_all = "*" in self.keywords_to_monitor
        self.left_strip_character = left_strip_character

    def run(self, **unused_kwargs: Any) -> None:
        for name, parameter in parameter_dict(self.model).items():
            # check if the parameter name matches any of the specified keywords
            if self.summarize_all or any(keyword in name for keyword in self.keywords_to_monitor):
                # keys are sometimes prepended with a character, which we strip
                name = name.lstrip(self.left_strip_character)
                self._summarize_parameter(name, parameter)

    def _summarize_parameter(self, name: str, param: Union[Parameter, tf.Variable]) -> None:
        """
        :param name: identifier used in tensorboard
        :param param: parameter to be stored in tensorboard
        """
        param = tf.reshape(param, (-1,))
        size = param.shape[0]

        if not isinstance(size, int):
            raise ValueError(
                f"The monitoring can not be autographed as the size of a parameter {param} "
                "is unknown at compile time. If compiling the monitor task is important, "
                "make sure the shape of all parameters is known beforehand. Otherwise, "
                "run the monitor outside the `tf.function`."
            )
        if size == 1:
            # if there's only one element do not add a numbered suffix
            tf.summary.scalar(name, param[0], step=self.current_step)
        else:
            it = range(size) if self.max_size == -1 else range(min(size, self.max_size))
            for i in it:
                tf.summary.scalar(f"{name}[{i}]", param[i], step=self.current_step)

