class TensorboardLogger(BaseLogger):
    """
    TensorBoard handler to log metrics, model/optimizer parameters, gradients during the training and validation.

    By default, this class favors `tensorboardX <https://github.com/lanpa/tensorboardX>`_ package if installed:

    .. code-block:: bash

        pip install tensorboardX

    otherwise, it falls back to using
    `PyTorch's SummaryWriter
    <https://pytorch.org/docs/stable/tensorboard.html>`_
    (>=v1.2.0).

    Args:
        args: Positional arguments accepted from
            `SummaryWriter
            <https://pytorch.org/docs/stable/tensorboard.html>`_.
        kwargs: Keyword arguments accepted from
            `SummaryWriter
            <https://pytorch.org/docs/stable/tensorboard.html>`_.
            For example, `log_dir` to setup path to the directory where to log.

    Examples:
        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log training loss at each iteration
            tb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss}
            )

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            tb_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)),
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            tb_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name='lr'  # optional
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model)
            )

            # Attach the logger to the trainer to log model's weights as a histogram after each epoch
            tb_logger.attach(
                trainer,
                event_name=Events.EPOCH_COMPLETED,
                log_handler=WeightsHistHandler(model)
            )

            # Attach the logger to the trainer to log model's gradients norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model)
            )

            # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
            tb_logger.attach(
                trainer,
                event_name=Events.EPOCH_COMPLETED,
                log_handler=GradsHistHandler(model)
            )

            # We need to close the logger when we are done
            tb_logger.close()

        It is also possible to use the logger as context manager:

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            with TensorboardLogger(log_dir="experiments/tb_logs") as tb_logger:

                trainer = Engine(update_fn)
                # Attach the logger to the trainer to log training loss at each iteration
                tb_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="training",
                    output_transform=lambda loss: {"loss": loss}
                )

    """

    def __init__(self, *args: Any, **kwargs: Any):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ModuleNotFoundError(
                    "This contrib module requires either tensorboardX or torch >= 1.2.0. "
                    "You may install tensorboardX with command: \n pip install tensorboardX \n"
                    "or upgrade PyTorch using your package manager of choice (pip or conda)."
                )

        self.writer = SummaryWriter(*args, **kwargs)

    def __getattr__(self, attr: Any) -> Any:
        return getattr(self.writer, attr)

    def close(self) -> None:
        self.writer.close()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)

class GradsHistHandler(BaseWeightsHandler):
    """Helper handler to log model's gradients as histograms.

    Args:
        model: model to log weights
        tag: common title for all produced plots. For example, "generator"
        whitelist: specific gradients to log. Should be list of model's submodules
            or parameters names, or a callable which gets weight along with its name
            and determines if its gradient should be logged. Names should be
            fully-qualified. For more information please refer to `PyTorch docs
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule>`_.
            If not given, all of model's gradients are logged.

    Examples:
        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model)
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log gradient of `fc.bias`
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model, whitelist=['fc.bias'])
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log gradient of weights which have shape (2, 1)
            def has_shape_2_1(n, p):
                return p.shape == (2,1)

            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model, whitelist=has_shape_2_1)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'GradsHistHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            logger.writer.add_histogram(
                tag=f"{tag_prefix}grads/{name}", values=p.grad.cpu().numpy(), global_step=global_step
            )

class WeightsHistHandler(BaseWeightsHandler):
    """Helper handler to log model's weights as histograms.

    Args:
        model: model to log weights
        tag: common title for all produced plots. For example, "generator"
        whitelist: specific weights to log. Should be list of model's submodules
            or parameters names, or a callable which gets weight along with its name
            and determines if it should be logged. Names should be fully-qualified.
            For more information please refer to `PyTorch docs
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule>`_.
            If not given, all of model's weights are logged.

    Examples:
        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model)
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log weights of `fc` layer
            weights = ['fc']

            # Attach the logger to the trainer to log weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model, whitelist=weights)
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log weights which name include 'conv'.
            weight_selector = lambda name, p: 'conv' in name

            # Attach the logger to the trainer to log weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model, whitelist=weight_selector)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsHistHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            name = name.replace(".", "/")
            logger.writer.add_histogram(
                tag=f"{tag_prefix}weights/{name}", values=p.data.cpu().numpy(), global_step=global_step
            )

