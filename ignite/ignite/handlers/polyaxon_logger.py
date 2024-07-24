class PolyaxonLogger(BaseLogger):
    """
    `Polyaxon tracking client <https://polyaxon.com/>`_ handler to log parameters and metrics during the training
    and validation.

    This class requires `polyaxon <https://github.com/polyaxon/polyaxon/>`_ package to be installed:

    .. code-block:: bash

        pip install polyaxon

        // If you are using polyaxon v0.x

        pip install polyaxon-client

    Args:
        args: Positional arguments accepted from
            `Experiment <https://polyaxon.com/docs/experimentation/tracking/client/>`_.
        kwargs: Keyword arguments accepted from
            `Experiment <https://polyaxon.com/docs/experimentation/tracking/client/>`_.

    Examples:
        .. code-block:: python

            from ignite.handlers.polyaxon_logger import *

            # Create a logger
            plx_logger = PolyaxonLogger()

            # Log experiment parameters:
            plx_logger.log_inputs(**{
                "seed": seed,
                "batch_size": batch_size,
                "model": model.__class__.__name__,

                "pytorch version": torch.__version__,
                "ignite version": ignite.__version__,
                "cuda version": torch.version.cuda,
                "device name": torch.cuda.get_device_name(0)
            })

            # Attach the logger to the trainer to log training loss at each iteration
            plx_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss}
            )

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            plx_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            plx_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)),
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            plx_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name='lr'  # optional
            )
            # to manually end a run
            plx_logger.close()
    """

    def __init__(self, *args: Any, **kwargs: Any):
        try:
            from polyaxon.tracking import Run

            self.experiment = Run(*args, **kwargs)

        except ImportError:
            try:
                from polyaxon_client.tracking import Experiment

                self.experiment = Experiment(*args, **kwargs)
            except ImportError:
                raise ModuleNotFoundError(
                    "This contrib module requires polyaxon to be installed.\n"
                    "For Polyaxon v1.x please install it with command: \n pip install polyaxon\n"
                    "For Polyaxon v0.x please install it with command: \n pip install polyaxon-client"
                )

    def close(self) -> None:
        try:
            self.experiment.end()
        except:
            pass

    def __getattr__(self, attr: Any) -> Any:
        return getattr(self.experiment, attr)

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)

