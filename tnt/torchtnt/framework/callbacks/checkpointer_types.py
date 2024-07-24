class KnobOptions:
    """
    Controls the knobs for Checkpoints.

    Args:
        max_per_rank_io_concurrency: Maximum number of concurrent IO operations per rank in checkpointing.
                                     Defaults to 16.
        enable_storage_optimization: Enable storage efficiency optimizations for Distributed Checkpointing.
    """

    # use a more conservative number of concurrent IO operations per rank in Checkpointing
    # the default value of 16 is too bandwidth hungry for most users
    max_per_rank_io_concurrency: Optional[int] = None
    # This is a no-op and for future use. This would enable storage efficiency optimizations:
    # e.g. Compression, Batching, Quantization etc.
    enable_storage_optimization: bool = False

class RestoreOptions:
    """
    Options when restoring a snapshot.

    Args:
        restore_train_progress: Whether to restore the training progress state.
        restore_eval_progress: Whether to restore the evaluation progress state.
        restore_optimizers: Whether to restore the optimizer states.
        restore_lr_schedulers: Whether to restore the lr scheduler states.
        strict: Whether to strictly restore app state and the module state dict.
    """

    restore_train_progress: bool = True
    restore_eval_progress: bool = True
    restore_optimizers: bool = True
    restore_lr_schedulers: bool = True
    strict: bool = True

