def set_bypass_mode(cls, bypass: bool) -> None:
        """
        Set ``clearml.Task`` to offline mode.
        Will bypass all outside communication, and will save all data and logs to a local session folder.
        Should only be used in "standalone mode", when there is no access to the *clearml-server*.

        Args:
            bypass: If ``True``, all outside communication is skipped.
                Data and logs will be stored in a local session folder.
                For more information, please refer to `ClearML docs
                <https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/#offline-mode>`_.
        """
        from clearml import Task

        setattr(cls, "_bypass", bypass)
        Task.set_offline(offline_mode=bypass)

class _CallbacksContext:
        def __init__(
            self,
            callback_type: Type[Enum],
            slots: List,
            checkpoint_key: str,
            filename: str,
            basename: str,
            metadata: Optional[Mapping] = None,
        ) -> None:
            self._callback_type = callback_type
            self._slots = slots
            self._checkpoint_key = str(checkpoint_key)
            self._filename = filename
            self._basename = basename
            self._metadata = metadata

        def pre_callback(self, action: str, model_info: Any) -> Any:
            if action != self._callback_type.save:  # type: ignore[attr-defined]
                return model_info

            try:
                slot = self._slots.index(None)
                self._slots[slot] = model_info.upload_filename
            except ValueError:
                self._slots.append(model_info.upload_filename)
                slot = len(self._slots) - 1

            model_info.upload_filename = f"{self._basename}_{slot}{os.path.splitext(self._filename)[1]}"
            model_info.local_model_id = f"{self._checkpoint_key}:{model_info.upload_filename}"
            return model_info

        def post_callback(self, action: str, model_info: Any) -> Any:
            if action != self._callback_type.save:  # type: ignore[attr-defined]
                return model_info

            model_info.model.name = f"{model_info.task.name}: {self._filename}"
            prefix = "Checkpoint Metadata: "
            metadata_items = ", ".join(f"{k}={v}" for k, v in self._metadata.items()) if self._metadata else "none"
            metadata = f"{prefix}{metadata_items}"
            comment = "\n".join(
                metadata if line.startswith(prefix) else line for line in (model_info.model.comment or "").split("\n")
            )
            if prefix not in comment:
                comment += "\n" + metadata
            model_info.model.comment = comment

            return model_info

class ClearMLLogger(BaseLogger):
    """
    `ClearML <https://github.com/allegroai/clearml>`_ handler to log metrics, text, model/optimizer parameters,
    plots during training and validation.
    Also supports model checkpoints logging and upload to the storage solution of your choice (i.e. ClearML File server,
    S3 bucket etc.)

    .. code-block:: bash

        pip install clearml
        clearml-init

    Args:
        kwargs: Keyword arguments accepted from ``Task.init`` method.
            All arguments are optional. If a ClearML Task has already been created,
            kwargs will be ignored and the current ClearML Task will be used.

    Examples:
        .. code-block:: python

            from ignite.handlers.clearml_logger import *

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Attach the logger to the trainer to log training loss at each iteration
            clearml_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss}
            )

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            clearml_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            clearml_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)),
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            clearml_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name='lr'  # optional
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model)
            )

    """

    def __init__(self, **kwargs: Any):
        try:
            from clearml import Task
            from clearml.binding.frameworks.tensorflow_bind import WeightsGradientHistHelper
        except ImportError:
            raise ModuleNotFoundError(
                "This contrib module requires clearml to be installed. "
                "You may install clearml using: \n pip install clearml \n"
            )

        experiment_kwargs = {k: v for k, v in kwargs.items() if k not in ("project_name", "task_name", "task_type")}

        if self.bypass_mode():
            warnings.warn("ClearMLSaver: running in bypass mode")

        # Try to retrieve current the ClearML Task before trying to create a new one
        self._task = Task.current_task()

        if self._task is None:
            self._task = Task.init(
                project_name=kwargs.get("project_name"),
                task_name=kwargs.get("task_name"),
                task_type=kwargs.get("task_type", Task.TaskTypes.training),
                **experiment_kwargs,
            )

        self.clearml_logger = self._task.get_logger()

        self.grad_helper = WeightsGradientHistHelper(logger=self.clearml_logger, report_freq=1)

    @classmethod
    def set_bypass_mode(cls, bypass: bool) -> None:
        """
        Set ``clearml.Task`` to offline mode.
        Will bypass all outside communication, and will save all data and logs to a local session folder.
        Should only be used in "standalone mode", when there is no access to the *clearml-server*.

        Args:
            bypass: If ``True``, all outside communication is skipped.
                Data and logs will be stored in a local session folder.
                For more information, please refer to `ClearML docs
                <https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/#offline-mode>`_.
        """
        from clearml import Task

        setattr(cls, "_bypass", bypass)
        Task.set_offline(offline_mode=bypass)

    @classmethod
    def bypass_mode(cls) -> bool:
        """
        Returns the bypass mode state.

        Note:
            `GITHUB_ACTIONS` env will automatically set bypass_mode to ``True``
            unless overridden specifically with ``ClearMLLogger.set_bypass_mode(False)``.
            For more information, please refer to `ClearML docs
            <https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/#offline-mode>`_.

        Return:
            If True, ``clearml.Task`` is on offline mode, and all outside communication is skipped.
        """
        return getattr(cls, "_bypass", bool(os.environ.get("CI")))

    def __getattr__(self, attr: Any) -> Any:
        """
        Calls the corresponding method of ``clearml.Logger``.

        Args:
            attr: methods of the ``clearml.Logger`` class.
        """
        return getattr(self.clearml_logger, attr)

    def get_task(self) -> Any:
        """
        Returns the task context that the logger is reporting.

        Return:
            Returns the current task, equivalent to ``clearml.Task.current_task()``.
        """
        return self._task

    def close(self) -> None:
        self.clearml_logger.flush()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)

class GradsHistHandler(BaseWeightsHandler):
    """Helper handler to log model's gradients as histograms.

    Args:
        model: model to log weights
        tag: common title for all produced plots. For example, 'generator'
        whitelist: specific gradients to log. Should be list of model's submodules
            or parameters names, or a callable which gets weight along with its name
            and determines if its gradient should be logged. Names should be
            fully-qualified. For more information please refer to `PyTorch docs
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule>`_.
            If not given, all of model's gradients are logged.

    Examples:
        .. code-block:: python

            from ignite.handlers.clearml_logger import *

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model)
            )

        .. code-block:: python

            from ignite.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log gradient of `fc.bias`
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model, whitelist=['fc.bias'])
            )

        .. code-block:: python

            from ignite.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log gradient of weights which have shape (2, 1)
            def has_shape_2_1(n, p):
                return p.shape == (2,1)

            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model, whitelist=has_shape_2_1)
            )

    ..  versionchanged:: 0.4.9
            optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: ClearMLLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, ClearMLLogger):
            raise RuntimeError("Handler 'GradsHistHandler' works only with ClearMLLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            if p.grad is None:
                continue

            title_name, _, series_name = name.partition(".")
            logger.grad_helper.add_histogram(
                title=f"{tag_prefix}grads_{title_name}",
                series=series_name,
                step=global_step,
                hist_data=p.grad.cpu().numpy(),
            )

class WeightsHistHandler(BaseWeightsHandler):
    """Helper handler to log model's weights as histograms.

    Args:
        model: model to log weights
        tag: common title for all produced plots. For example, 'generator'
        whitelist: specific weights to log. Should be list of model's submodules
            or parameters names, or a callable which gets weight along with its name
            and determines if it should be logged. Names should be fully-qualified.
            For more information please refer to `PyTorch docs
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule>`_.
            If not given, all of model's weights are logged.

    Examples:
        .. code-block:: python

            from ignite.handlers.clearml_logger import *

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model)
            )

        .. code-block:: python

            from ignite.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log weights of `fc` layer
            weights = ['fc']

            # Attach the logger to the trainer to log weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model, whitelist=weights)
            )

        .. code-block:: python

            from ignite.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log weights which name include 'conv'.
            weight_selector = lambda name, p: 'conv' in name

            # Attach the logger to the trainer to log weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model, whitelist=weight_selector)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: ClearMLLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, ClearMLLogger):
            raise RuntimeError("Handler 'WeightsHistHandler' works only with ClearMLLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            title_name, _, series_name = name.partition(".")

            logger.grad_helper.add_histogram(
                title=f"{tag_prefix}weights_{title_name}",
                series=series_name,
                step=global_step,
                hist_data=p.data.cpu().numpy(),
            )

