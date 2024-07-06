def deprecated(
    deprecated_in: str, removed_in: str = "", reasons: Tuple[str, ...] = (), raise_exception: bool = False
) -> Callable:
    F = TypeVar("F", bound=Callable[..., Any])

    def decorator(func: F) -> F:
        func_doc = func.__doc__ if func.__doc__ else ""
        deprecation_warning = (
            f"This function has been deprecated since version {deprecated_in}"
            + (f" and will be removed in version {removed_in}" if removed_in else "")
            + ".\n Please refer to the documentation for more details."
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Dict[str, Any]) -> Callable:
            if raise_exception:
                raise DeprecationWarning(deprecation_warning)
            warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        appended_doc = f".. deprecated:: {deprecated_in}" + ("\n\n\t" if len(reasons) > 0 else "")

        for reason in reasons:
            appended_doc += "\n\t- " + reason
        wrapper.__doc__ = f"**Deprecated function**.\n\n    {func_doc}{appended_doc}"
        return cast(F, wrapper)

    return decorator

def manual_seed(seed: int) -> None:
    """Setup random state from a seed for `torch`, `random` and optionally `numpy` (if can be imported).

    Args:
        seed: Random state seed

    .. versionchanged:: 0.4.3
        Added ``torch.cuda.manual_seed_all(seed)``.

    .. versionchanged:: 0.4.5
        Added ``torch_xla.core.xla_model.set_rng_state(seed)``.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(seed)
    except ImportError:
        pass

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

def setup_logger(
    name: Optional[str] = "ignite",
    level: int = logging.INFO,
    stream: Optional[TextIO] = None,
    format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
    filepath: Optional[str] = None,
    distributed_rank: Optional[int] = None,
    reset: bool = False,
    encoding: Optional[str] = "utf-8",
) -> logging.Logger:
    """Setups logger: name, level, format etc.

    Args:
        name: new name for the logger. If None, the standard logger is used.
        level: logging level, e.g. CRITICAL, ERROR, WARNING, INFO, DEBUG.
        stream: logging stream. If None, the standard stream is used (sys.stderr).
        format: logging format. By default, `%(asctime)s %(name)s %(levelname)s: %(message)s`.
        filepath: Optional logging file path. If not None, logs are written to the file.
        distributed_rank: Optional, rank in distributed configuration to avoid logger setup for workers.
            If None, distributed_rank is initialized to the rank of process.
        reset: if True, reset an existing logger rather than keep format, handlers, and level.
        encoding: open the file with the encoding. By default, 'utf-8'.

    Returns:
        logging.Logger

    Examples:
        Improve logs readability when training with a trainer and evaluator:

        .. code-block:: python

            from ignite.utils import setup_logger

            trainer = ...
            evaluator = ...

            trainer.logger = setup_logger("trainer")
            evaluator.logger = setup_logger("evaluator")

            trainer.run(data, max_epochs=10)

            # Logs will look like
            # 2020-01-21 12:46:07,356 trainer INFO: Engine run starting with max_epochs=5.
            # 2020-01-21 12:46:07,358 trainer INFO: Epoch[1] Complete. Time taken: 00:5:23
            # 2020-01-21 12:46:07,358 evaluator INFO: Engine run starting with max_epochs=1.
            # 2020-01-21 12:46:07,358 evaluator INFO: Epoch[1] Complete. Time taken: 00:01:02
            # ...

        Every existing logger can be reset if needed

        .. code-block:: python

            logger = setup_logger(name="my-logger", format="=== %(name)s %(message)s")
            logger.info("first message")
            setup_logger(name="my-logger", format="+++ %(name)s %(message)s", reset=True)
            logger.info("second message")

            # Logs will look like
            # === my-logger first message
            # +++ my-logger second message

        Change the level of an existing internal logger

        .. code-block:: python

            setup_logger(
                name="ignite.distributed.launcher.Parallel",
                level=logging.WARNING
            )

    .. versionchanged:: 0.4.3
        Added ``stream`` parameter.

    .. versionchanged:: 0.4.5
        Added ``reset`` parameter.

    .. versionchanged:: 0.5.1
        Argument ``encoding`` added to correctly handle special characters in the file, default "utf-8".
    """
    # check if the logger already exists
    existing = name is None or name in logging.root.manager.loggerDict

    # if existing, get the logger otherwise create a new one
    logger = logging.getLogger(name)

    if distributed_rank is None:
        import ignite.distributed as idist

        distributed_rank = idist.get_rank()

    # Remove previous handlers
    if distributed_rank > 0 or reset:
        if logger.hasHandlers():
            for h in list(logger.handlers):
                logger.removeHandler(h)

    if distributed_rank > 0:
        # Add null handler to avoid multiple parallel messages
        logger.addHandler(logging.NullHandler())

    # Keep the existing configuration if not reset
    if existing and not reset:
        return logger

    if distributed_rank == 0:
        logger.setLevel(level)

        formatter = logging.Formatter(format)

        ch = logging.StreamHandler(stream=stream)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if filepath is not None:
            fh = logging.FileHandler(filepath, encoding=encoding)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    # don't propagate to ancestors
    # the problem here is to attach handlers to loggers
    # should we provide a default configuration less open ?
    if name is not None:
        logger.propagate = False

    return logger

