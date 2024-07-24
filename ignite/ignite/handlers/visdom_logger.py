class VisdomLogger(BaseLogger):
    """
    VisdomLogger handler to log metrics, model/optimizer parameters, gradients during the training and validation.

    This class requires `visdom <https://github.com/fossasia/visdom/>`_ package to be installed:

    .. code-block:: bash


        pip install git+https://github.com/fossasia/visdom.git

    Args:
        server: visdom server URL. It can be also specified by environment variable `VISDOM_SERVER_URL`
        port: visdom server's port. It can be also specified by environment variable `VISDOM_PORT`
        num_workers: number of workers to use in `concurrent.futures.ThreadPoolExecutor` to post data to
            visdom server. Default, `num_workers=1`. If `num_workers=0` and logger uses the main thread. If using
            Python 2.7 and `num_workers>0` the package `futures` should be installed: `pip install futures`
        kwargs: kwargs to pass into
            `visdom.Visdom <https://github.com/fossasia/visdom#visdom-arguments-python-only>`_.

    Note:
        We can also specify username/password using environment variables: VISDOM_USERNAME, VISDOM_PASSWORD


    .. warning::

        Frequent logging, e.g. when logger is attached to `Events.ITERATION_COMPLETED`, can slow down the run if the
        main thread is used to send the data to visdom server (`num_workers=0`). To avoid this situation we can either
        log less frequently or set `num_workers=1`.

    Examples:
        .. code-block:: python

            from ignite.handlers.visdom_logger import *

            # Create a logger
            vd_logger = VisdomLogger()

            # Attach the logger to the trainer to log training loss at each iteration
            vd_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss}
            )

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            vd_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            vd_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)),
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            vd_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name='lr'  # optional
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            vd_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model)
            )

            # Attach the logger to the trainer to log model's gradients norm after each iteration
            vd_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model)
            )

            # We need to close the logger with we are done
            vd_logger.close()

        It is also possible to use the logger as context manager:

        .. code-block:: python

            from ignite.handlers.visdom_logger import *

            with VisdomLogger() as vd_logger:

                trainer = Engine(update_fn)
                # Attach the logger to the trainer to log training loss at each iteration
                vd_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="training",
                    output_transform=lambda loss: {"loss": loss}
                )

    .. versionchanged:: 0.4.7
        accepts an optional list of `state_attributes`
    """

    def __init__(
        self,
        server: Optional[str] = None,
        port: Optional[int] = None,
        num_workers: int = 1,
        raise_exceptions: bool = True,
        **kwargs: Any,
    ):
        try:
            import visdom
        except ImportError:
            raise ModuleNotFoundError(
                "This contrib module requires visdom package. "
                "Please install it with command:\n"
                "pip install git+https://github.com/fossasia/visdom.git"
            )

        if num_workers > 0:
            # If visdom is installed, one of its dependencies `tornado`
            # requires also `futures` to be installed.
            # Let's check anyway if we can import it.
            try:
                from concurrent.futures import ThreadPoolExecutor
            except ImportError:
                raise ModuleNotFoundError(
                    "This contrib module requires concurrent.futures module"
                    "Please install it with command:\n"
                    "pip install futures"
                )

        if server is None:
            server = cast(str, os.environ.get("VISDOM_SERVER_URL", "localhost"))

        if port is None:
            port = int(os.environ.get("VISDOM_PORT", 8097))

        if "username" not in kwargs:
            username = os.environ.get("VISDOM_USERNAME", None)
            kwargs["username"] = username

        if "password" not in kwargs:
            password = os.environ.get("VISDOM_PASSWORD", None)
            kwargs["password"] = password

        self.vis = visdom.Visdom(server=server, port=port, raise_exceptions=raise_exceptions, **kwargs)

        if not self.vis.offline and not self.vis.check_connection():  # type: ignore[attr-defined]
            raise RuntimeError(f"Failed to connect to Visdom server at {server}. Did you run python -m visdom.server ?")

        self.executor: Union[_DummyExecutor, "ThreadPoolExecutor"] = _DummyExecutor()
        if num_workers > 0:
            self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def _save(self) -> None:
        self.vis.save([self.vis.env])  # type: ignore[attr-defined]

    def close(self) -> None:
        self.executor.shutdown()
        self.vis.close()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)

