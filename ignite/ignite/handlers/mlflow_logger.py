class MLflowLogger(BaseLogger):
    """
    `MLflow <https://mlflow.org>`_ tracking client handler to log parameters and metrics during the training
    and validation.

    This class requires `mlflow package <https://github.com/mlflow/mlflow/>`_ to be installed:

    .. code-block:: bash

        pip install mlflow

    Args:
        tracking_uri: MLflow tracking uri. See MLflow docs for more details

    Examples:
        .. code-block:: python

            from ignite.handlers.mlflow_logger import *

            # Create a logger
            mlflow_logger = MLflowLogger()

            # Log experiment parameters:
            mlflow_logger.log_params({
                "seed": seed,
                "batch_size": batch_size,
                "model": model.__class__.__name__,

                "pytorch version": torch.__version__,
                "ignite version": ignite.__version__,
                "cuda version": torch.version.cuda,
                "device name": torch.cuda.get_device_name(0)
            })

            # Attach the logger to the trainer to log training loss at each iteration
            mlflow_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {'loss': loss}
            )

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            mlflow_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            mlflow_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)),
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            mlflow_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name='lr'  # optional
            )
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        try:
            import mlflow
        except ImportError:
            raise ModuleNotFoundError(
                "This contrib module requires mlflow to be installed. "
                "Please install it with command: \n pip install mlflow"
            )

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        self.active_run = mlflow.active_run()
        if self.active_run is None:
            self.active_run = mlflow.start_run()

    def __getattr__(self, attr: Any) -> Any:
        import mlflow

        return getattr(mlflow, attr)

    def close(self) -> None:
        import mlflow

        mlflow.end_run()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)

