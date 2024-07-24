class MovementSchedulerParams:
    """
    Stores the params to initialize the scheduler of movement sparsity.
    """

    def __init__(
        self,
        warmup_start_epoch: int,
        warmup_end_epoch: int,
        importance_regularization_factor: float,
        enable_structured_masking: bool = MOVEMENT_ENABLE_STRUCTURED_MASKING,
        init_importance_threshold: Optional[float] = None,
        final_importance_threshold: float = MOVEMENT_FINAL_IMPORTANCE_THRESHOLD,
        power: float = MOVEMENT_POWER,
        steps_per_epoch: Optional[int] = None,
    ):
        """
        Initializes and validates the params for scheduler.

        :param warmup_start_epoch: Index of the starting epoch (inclusive) for warmup stage.
        :param warmup_end_epoch: Index of the end epoch (exclusive) for warmup stage.
        :param importance_regularization_factor: The regularization factor on weight importance scores.
        :param enable_structured_masking: Whether to do structured mask resolution after warmup stage.
        :param init_importance_threshold: The initial value of importance threshold during warmup stage.
        :param final_importance_threshold: The final value of importance threshold during warmup stage.
        :param power: The power value of polynomial decay for threshold update during warmup stage.
        :param steps_per_epoch: Number of training steps in one epoch.
        """

        if steps_per_epoch is None and warmup_start_epoch < 1:
            raise ValueError(
                "`warmup_start_epoch` must be >= 1 to enable the auto calculation of "
                "`steps_per_epoch`. Please either change `warmup_start_epoch` to a larger "
                "number or specify `steps_per_epoch` in the config."
            )

        if warmup_start_epoch < 0 or warmup_end_epoch <= warmup_start_epoch:
            raise ValueError("Movement sparsity requires 0 <= warmup_start_epoch < warmup_end_epoch.")

        if importance_regularization_factor < 0:
            raise ValueError("`importance_regularization_factor` should not be a negative number.")

        if init_importance_threshold is not None and init_importance_threshold >= final_importance_threshold:
            nncf_logger.warning(
                "`init_importance_threshold` is equal to or greater than "
                "`final_importance_threshold`. Movement sparsity may not work as expected."
            )

        self.warmup_start_epoch = warmup_start_epoch
        self.warmup_end_epoch = warmup_end_epoch
        self.importance_regularization_factor = importance_regularization_factor
        self.enable_structured_masking = enable_structured_masking
        self.init_importance_threshold = init_importance_threshold
        self.final_importance_threshold = final_importance_threshold
        self.power = power
        self.steps_per_epoch = steps_per_epoch

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "MovementSchedulerParams":
        """
        Initialize `MovementSchedulerParams` object from the config in dict format.

        :param params: A dict that specifies the parameters of movement sparsity scheduler.
        :return: A `MovementSchedulerParams` object that stores the parameters from `params`.
        """
        warmup_start_epoch: int = params.get("warmup_start_epoch")
        warmup_end_epoch: int = params.get("warmup_end_epoch")
        importance_regularization_factor: float = params.get("importance_regularization_factor")
        enable_structured_masking: bool = params.get("enable_structured_masking", MOVEMENT_ENABLE_STRUCTURED_MASKING)
        init_importance_threshold: Optional[float] = params.get("init_importance_threshold")
        final_importance_threshold: float = params.get(
            "final_importance_threshold", MOVEMENT_FINAL_IMPORTANCE_THRESHOLD
        )
        power: float = params.get("power", MOVEMENT_POWER)
        steps_per_epoch: Optional[int] = params.get("steps_per_epoch")

        if None in [warmup_start_epoch, warmup_end_epoch, importance_regularization_factor]:
            raise ValueError(
                "`warmup_start_epoch`, `warmup_start_epoch` and `importance_regularization_factor` "
                "are required in config for Movement Sparsity."
            )

        return cls(
            warmup_start_epoch=warmup_start_epoch,
            warmup_end_epoch=warmup_end_epoch,
            importance_regularization_factor=importance_regularization_factor,
            enable_structured_masking=enable_structured_masking,
            init_importance_threshold=init_importance_threshold,
            final_importance_threshold=final_importance_threshold,
            power=power,
            steps_per_epoch=steps_per_epoch,
        )

