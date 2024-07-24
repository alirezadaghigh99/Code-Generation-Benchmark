class NeptuneSaver(BaseSaveHandler):
    """Handler that saves input checkpoint to the Neptune server.

    Args:
        neptune_logger: an instance of
            NeptuneLogger class.

    .. Note ::

        NeptuneSaver is currently not supported on Windows.

    Examples:
        .. code-block:: python

            from ignite.handlers.neptune_logger import *

            # Create a logger
            # We are using the api_token for the anonymous user neptuner but you can use your own.

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite","minst"] # Optional
            )

            ...
            evaluator = create_supervised_evaluator(model, metrics=metrics, ...)
            ...

            from ignite.handlers import Checkpoint

            def score_function(engine):
                return engine.state.metrics["accuracy"]

            to_save = {"model": model}

            # pass neptune logger to NeptuneServer

            handler = Checkpoint(
                to_save,
                NeptuneSaver(npt_logger), n_saved=2,
                filename_prefix="best", score_function=score_function,
                score_name="validation_accuracy",
                global_step_transform=global_step_from_engine(trainer)
            )

            evaluator.add_event_handler(Events.COMPLETED, handler)

            # We need to close the logger when we are done
            npt_logger.close()

    For example, you can access model checkpoints and download them from here:
    https://ui.neptune.ai/o/shared/org/pytorch-ignite-integration/e/PYTOR1-18/charts

    """

    @idist.one_rank_only()
    def __init__(self, neptune_logger: NeptuneLogger):
        self._logger = neptune_logger

    @idist.one_rank_only()
    def __call__(self, checkpoint: Mapping, filename: str, metadata: Optional[Mapping] = None) -> None:
        # wont work on XLA

        # Imports for BC compatibility
        try:
            # neptune-client<1.0.0 package structure
            with warnings.catch_warnings():
                # ignore the deprecation warnings
                warnings.simplefilter("ignore")
                from neptune.new.types import File
        except ImportError:
            # neptune>=1.0.0 package structure
            from neptune.types import File

        with tempfile.NamedTemporaryFile() as tmp:
            # we can not use tmp.name to open tmp.file twice on Win32
            # https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
            torch.save(checkpoint, tmp.file)

            # rewind the buffer
            tmp.file.seek(0)

            # hold onto the file stream for uploading.
            # NOTE: This won't load the whole file in memory and upload
            #       the stream in smaller chunks.
            self._logger[filename].upload(File.from_stream(tmp.file))

    @idist.one_rank_only(with_barrier=True)
    def remove(self, filename: str) -> None:
        del self._logger.experiment[filename]

class NeptuneLogger(BaseLogger):
    """
    `Neptune <https://neptune.ai/>`_ handler to log metrics, model/optimizer parameters and gradients during training
    and validation. It can also log model checkpoints to Neptune.

    .. code-block:: bash

        pip install neptune

    Args:
        api_token: Neptune API token, found on https://neptune.ai -> User menu -> "Get your API token".
           If None, the value of the NEPTUNE_API_TOKEN environment variable is used. To keep your token
           secure, you should set it to the environment variable rather than including it in your code.
        project: Name of a Neptune project, in the form "workspace-name/project-name".
           For example "tom/mnist-classification".
           If None, the value of the NEPTUNE_PROJECT environment variable is used.
        **kwargs: Other arguments to be passed to the `init_run()` function.

    Examples:
        .. code-block:: python

            from ignite.handlers.neptune_logger import *

            # Create a logger
            # Note: We are using the API token for anonymous logging. You can pass your own token, or save it as an
            # environment variable and leave out the api_token argument.

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project="common/pytorch-ignite-integration",
                name="cnn-mnist",  # Optional,
                tags=["pytorch-ignite", "minst"],  # Optional
            )

            # Attach the logger to the trainer to log training loss at each iteration.
            npt_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss},
            )

            # Attach the logger to the evaluator on the training dataset and log NLL
            # and accuracy metrics after each epoch.
            # We set up `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            npt_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL and accuracy metrics after
            # each epoch. We set up `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            npt_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the trainer to log optimizer parameters, such as learning rate at each iteration.
            npt_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name="lr",  # optional
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration.
            npt_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model),
            )

        Explore runs with Neptune tracking here:
        https://app.neptune.ai/o/common/org/pytorch-ignite-integration/

        You can also save model checkpoints to a Neptune:

        .. code-block:: python

            from ignite.handlers import Checkpoint


            def score_function(engine):
                return engine.state.metrics["accuracy"]


            to_save = {"model": model}
            handler = Checkpoint(
                to_save,
                NeptuneSaver(npt_logger), n_saved=2,
                filename_prefix="best",
                score_function=score_function,
                score_name="validation_accuracy",
                global_step_transform=global_step_from_engine(trainer),
            )
            validation_evaluator.add_event_handler(Events.COMPLETED, handler)

        It is also possible to use the logger as a context manager:

        .. code-block:: python

            from ignite.handlers.neptune_logger import *

            with NeptuneLogger() as npt_logger:
                trainer = Engine(update_fn)
                # Attach the logger to the trainer to log training loss at each iteration
                npt_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="training",
                    output_transform=lambda loss: {"loss": loss},
                )

    """

    def __getattr__(self, attr: Any) -> Any:
        return getattr(self.experiment, attr)

    def __getitem__(self, key: str) -> Any:
        return self.experiment[key]

    def __setitem__(self, key: str, val: Any) -> Any:
        self.experiment[key] = val

    def __init__(self, api_token: Optional[str] = None, project: Optional[str] = None, **kwargs: Any) -> None:
        try:
            try:
                # neptune-client<1.0.0 package structure
                with warnings.catch_warnings():
                    # ignore the deprecation warnings
                    warnings.simplefilter("ignore")
                    import neptune.new as neptune
            except ImportError:
                # neptune>=1.0.0 package structure
                import neptune
        except ImportError:
            raise ModuleNotFoundError(
                "This contrib module requires the Neptune client library to be installed. "
                "Install neptune with the command: \n pip install neptune \n"
            )

        run = neptune.init_run(
            api_token=api_token,
            project=project,
            **kwargs,
        )
        from ignite import __version__

        run[_INTEGRATION_VERSION_KEY] = __version__

        self.experiment = run

    def close(self) -> None:
        self.experiment.stop()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)

