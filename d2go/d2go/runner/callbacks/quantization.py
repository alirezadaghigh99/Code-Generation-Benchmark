class QuantizationAwareTraining(Callback, QuantizationMixin):
    """Enable QAT of a model using the STL Trainer.

    Node that this callback makes changes during training in order to properly
    quantize the provided LightningModule.

    Example::
        >>> from stl.lightning.callbacks.quantization import QuantizationAwareTraining
        >>> from pytorch_lightning import Trainer
        >>> from stl.lightning.utilities.model import mode

        ...

        # MyLightningModule must define val_dataloader() which is used both for
        # validation as well as calibration of the quantized model.
        >>> model = MyLightningModule(...)
        >>> qat = QuantizationAwareTraining()
        >>> trainer = Trainer(
        ...    callbacks=[qat],
        ... )

        # This will convert the model into one that is quantizeable, train it,
        # and then quantize it after training is done.
        >>> trainer.fit(model)

        # You can use the model directly.
        >>> input = ...
        >>> with mode(model, training=False) as m:
        ...     quantized_out = m(input)

    If you only wish to quantize parts of your model, please see QuantizationMixin
    for an example of how to do this.

    Properties:
        transforms: A list of ModelTransform's applied to the model exactly once
            as specified during training. Example transforms are enabling/disabling
            observers/quants, which are added to this list based on the init
            parameters to this callback. Users can further augment the list
            with more custom modules.
        prepared: If set, this is the prepared model. Only available
            after .fit() starts.
        qconfig_dicts:
            This is a map from the `module_qualified_name` to the corresponding QConfigDict
            to apply to that module. For example, suppose your LightningModule contains
            two submodules module.scriptable and module.not_scriptable. You'd provide
            a qconfig_dicts like:
                {
                    "scriptable": ...
                }
            This will quantize just module.scriptable using the provided QConfigDict,
            or a default one. If you wish to quantize the entire LightningModule,
            simply use "" as the qualified name. The name should match the names
            returned by module.named_modules().
        quantized: If set, this is the fully quantized model. Only available
            after .fit() finishes.
    """

    def __init__(
        self,
        start_step: int = 0,
        enable_observer: Tuple[int, Optional[int]] = (0, None),
        freeze_bn_step: Optional[int] = None,
        qconfig_dicts: Optional[
            Dict[str, Optional[Dict[str, Union[QConfig, QConfigDynamic]]]]
        ] = None,
        preserved_attrs: Optional[List[str]] = None,
        skip_conversion: bool = False,
    ) -> None:
        """
        Args:
            start_step: The training step at which QAT is enabled. The model is
                always mutated with the appropriate stubs, but they are disabled
                until the start of this training step.
                See FakeQuantizeBase.fake_quant_enabled
            enable_observer: The half-open interval [a, b) in steps during which the
                observers are enabled. See FakeQuantizeBase.observer_enabled. If
                b is None, the observer is never disabled once enabled.
            freeze_bn_step: If specified, the step at which we apply freeze the
                collection of batch normalization layer statistics for QAT.
            qconfig_dicts: If given, used for quantization of the model during training.
            preserved_attrs: If provided, a list of attributes to preserve across
                quantized modules. These are preserved only if they already exists.
        """
        if start_step < 0:
            raise ValueError(
                f"The starting step of QAT must be non-negative. Got {start_step}."
            )
        start_observer, end_observer = enable_observer
        if start_observer < 0:
            raise ValueError(
                f"The starting step for the observer must be non-negative. Got {start_observer}."
            )
        if end_observer and end_observer <= start_observer:
            raise ValueError(
                f"The observation interval must contain at least one step. Got [{start_step}, {end_observer})."
            )
        if freeze_bn_step and freeze_bn_step < 0:
            raise ValueError(
                f"The step at which batch norm layers are frozen must be non-negative. Got {freeze_bn_step}."
            )
        self.transforms: List[ModelTransform] = []
        if start_step > 0:
            self.transforms.extend(
                [
                    # Enabled by default, so the assumption for > 0 is that the
                    # user wants it disabled then enabled.
                    ModelTransform(
                        fn=torch.ao.quantization.disable_fake_quant,
                        step=0,
                        message="Disable fake quant",
                    ),
                    ModelTransform(
                        fn=torch.ao.quantization.enable_fake_quant,
                        step=start_step,
                        message="Enable fake quant to start QAT",
                    ),
                ]
            )
        if start_observer > 0:
            self.transforms.extend(
                # See comment for start_step above.
                [
                    ModelTransform(
                        fn=torch.ao.quantization.disable_observer,
                        step=0,
                        message="Disable observer",
                    ),
                    ModelTransform(
                        fn=torch.ao.quantization.enable_observer,
                        step=start_observer,
                        message="Start observer",
                    ),
                ]
            )
        if end_observer is not None:
            self.transforms.append(
                ModelTransform(
                    fn=torch.ao.quantization.disable_observer,
                    step=end_observer,
                    message="End observer",
                )
            )
        if freeze_bn_step is not None:
            self.transforms.append(
                ModelTransform(
                    fn=torch.nn.intrinsic.qat.freeze_bn_stats,
                    step=freeze_bn_step,
                    message="Freeze BN",
                )
            )

        self.prepared: Optional[torch.nn.Module] = None
        self.preserved_attrs = set([] if preserved_attrs is None else preserved_attrs)
        if not qconfig_dicts:
            self.qconfig_dicts: QConfigDicts = {"": {"": get_default_qat_qconfig()}}
        else:
            self.qconfig_dicts: QConfigDicts = {
                key: value if value else {"": get_default_qat_qconfig()}
                for key, value in qconfig_dicts.items()
            }
        self.quantized: Optional[torch.nn.Module] = None
        self.skip_conversion = skip_conversion

    @classmethod
    def from_config(cls, cfg: CfgNode):
        qat = cfg.QUANTIZATION.QAT
        callback = cls(
            qconfig_dicts=(
                {submodule: None for submodule in cfg.QUANTIZATION.MODULES}
                if cfg.QUANTIZATION.MODULES
                else None
            ),
            # We explicitly pass this to maintain properties for now.
            preserved_attrs=["model.backbone.size_divisibility"],
            start_step=qat.START_ITER,
            enable_observer=(qat.ENABLE_OBSERVER_ITER, qat.DISABLE_OBSERVER_ITER),
            freeze_bn_step=qat.FREEZE_BN_ITER,
            skip_conversion=True,  # convert_fx will be handled by D2Go exporter
        )
        if qat.UPDATE_OBSERVER_STATS_PERIODICALLY:
            callback.transforms.append(
                ModelTransform(
                    interval=qat.UPDATE_OBSERVER_STATS_PERIOD,
                    fn=observer_update_stat,
                    message="Updating observers.",
                )
            )
        return callback

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Override the model with a quantized-aware version on setup.

        This is the earliest place we can override this model which allows for
        appropriate behavior when restoring from checkpoints, as well as connecting
        to accelerators, etc.

        The model is only prepared once.
        """
        # Only prepare the model once.
        if hasattr(pl_module, "_prepared"):
            return

        with mode(pl_module, training=True) as train:
            prepared = self.prepare(
                deepcopy(train),
                configs=self.qconfig_dicts,
                attrs=self.preserved_attrs,
            )
            # freeze the original model since only the prepared model will
            # participate in forward.
            for x in train.parameters():
                x.requires_grad = False
            pl_module._prepared = prepared
        pl_module.forward = MethodType(_quantized_forward, pl_module)
        self.prepared = pl_module._prepared

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Applies model transforms at as specified during training."""
        apply_only_once = []
        current_step = trainer.global_step
        for i, transform in enumerate(self.transforms):
            if (transform.step is not None and transform.step <= current_step) or (
                transform.interval is not None
                and current_step % transform.interval == 0
            ):
                self.prepared.apply(transform.fn)
                rank_zero_info(
                    f"[QAT] {transform.message} at step={trainer.global_step}."
                )
            if transform.step is not None and transform.step <= current_step:
                apply_only_once.append(i)

        if apply_only_once:
            self.transforms = [
                transform
                for i, transform in enumerate(self.transforms)
                if i not in set(apply_only_once)
            ]

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Quantize the weights since training has finalized."""
        if hasattr(pl_module, "_quantized") or self.skip_conversion:
            return
        pl_module._quantized = self.convert(
            pl_module._prepared, self.qconfig_dicts.keys(), attrs=self.preserved_attrs
        )
        self.quantized = pl_module._quantized

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Make sure we have a quantized version.

        This handles the edge case where a user does .test() without .fit() first.
        """
        if hasattr(pl_module, "_quantized"):
            return
        pl_module._quantized = self.convert(
            pl_module._prepared, self.qconfig_dicts.keys(), attrs=self.preserved_attrs
        )
        self.quantized = pl_module._quantized

class ModelTransform:
    """Defines a step or interval at which fn should be .apply(fn)'ed and a message to log.

    Properties:
        fn: The function to apply. Must be passable to torch.nn.Module.apply(fn).
        step: Only one of `step` or `interval` must be defined. If step is defined,
             `fn` will be applied exactly once right before `step` step begins.
        interval: Only one of `step` or `interval` must be defined. If `interval`
            is defined, the transform will be applied periodically every
            `interval` steps.
        message: A short non-punctuated message to log in the master worker when
        this transform is triggered.
    """

    fn: Callable[[torch.nn.Module], None]
    message: str
    step: Optional[int] = None
    interval: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate a few properties for early failure."""
        if (self.step is None and self.interval is None) or (
            self.step is not None and self.interval is not None
        ):
            raise TypeError("Exactly one of step or interval must be defined.")
        if self.step is not None and self.step < 0:
            raise ValueError("step must be non-negative.")
        if self.interval is not None and self.interval <= 0:
            raise ValueError("interval must be positive.")

