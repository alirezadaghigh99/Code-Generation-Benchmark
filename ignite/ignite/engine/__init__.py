def create_supervised_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable[[Any, Any], torch.Tensor], torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
    amp_mode: Optional[str] = None,
    scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
    gradient_accumulation_steps: int = 1,
    model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
) -> Engine:
    """Factory function for creating a trainer for supervised models.

    Args:
        model: the model to train.
        optimizer: the optimizer to use.
        loss_fn: the loss function that receives `y_pred` and `y`, and returns the loss as a tensor.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        model_transform: function that receives the output from the model and convert it into the form as required
            by the loss function
        output_transform: function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        deterministic: if True, returns deterministic engine of type
            :class:`~ignite.engine.deterministic.DeterministicEngine`, otherwise :class:`~ignite.engine.engine.Engine`
            (default: False).
        amp_mode: can be ``amp`` or ``apex``, model and optimizer will be casted to float16 using
            `torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`_ for ``amp`` and
            using `apex <https://nvidia.github.io/apex>`_ for ``apex``. (default: None)
        scaler: GradScaler instance for gradient scaling if `torch>=1.6.0`
            and ``amp_mode`` is ``amp``. If ``amp_mode`` is ``apex``, this argument will be ignored.
            If True, will create default GradScaler. If GradScaler instance is passed, it will be used instead.
            (default: False)
        gradient_accumulation_steps: Number of steps the gradients should be accumulated across.
            (default: 1 (means no gradient accumulation))
        model_fn: the model function that receives `model` and `x`, and returns `y_pred`.

    Returns:
        a trainer engine with supervised update function.

    Examples:

        Create a trainer

        .. code-block:: python

            from ignite.engine import create_supervised_trainer
            from ignite.utils import convert_tensor
            from ignite.handlers.tqdm_logger import ProgressBar

            model = ...
            loss = ...
            optimizer = ...
            dataloader = ...

            def prepare_batch_fn(batch, device, non_blocking):
                x = ...  # get x from batch
                y = ...  # get y from batch

                # return a tuple of (x, y) that can be directly runned as
                # `loss_fn(model(x), y)`
                return (
                    convert_tensor(x, device, non_blocking),
                    convert_tensor(y, device, non_blocking)
                )

            def output_transform_fn(x, y, y_pred, loss):
                # return only the loss is actually the default behavior for
                # trainer engine, but you can return anything you want
                return loss.item()

            trainer = create_supervised_trainer(
                model,
                optimizer,
                loss,
                prepare_batch=prepare_batch_fn,
                output_transform=output_transform_fn
            )

            pbar = ProgressBar()
            pbar.attach(trainer, output_transform=lambda x: {"loss": x})

            trainer.run(dataloader, max_epochs=5)

    Note:
        If ``scaler`` is True, GradScaler instance will be created internally and trainer state has attribute named
        ``scaler`` for that instance and can be used for saving and loading.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
        of the processed batch by default.

    .. warning::
        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.
        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_
        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    .. warning::
        If ``amp_mode='apex'`` , the model(s) and optimizer(s) must be initialized beforehand
        since ``amp.initialize`` should be called after you have finished constructing your model(s)
        and optimizer(s), but before you send your model through any DistributedDataParallel wrapper.

        See more: https://nvidia.github.io/apex/amp.html#module-apex.amp

    .. versionchanged:: 0.4.5

        - Added ``amp_mode`` argument for automatic mixed precision.
        - Added ``scaler`` argument for gradient scaling.

    .. versionchanged:: 0.4.7
        Added Gradient Accumulation argument for all supervised training methods.
    .. versionchanged:: 0.4.11
        Added ``model_transform`` to transform model's output
    .. versionchanged:: 0.4.13
        Added `model_fn` to customize model's application on the sample
    .. versionchanged:: 0.5.0
        Added support for ``mps`` device
    """

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    on_mps = "mps" in device_type if device_type is not None else False
    mode, _scaler = _check_arg(on_tpu, on_mps, amp_mode, scaler)

    if mode == "amp":
        _update = supervised_training_step_amp(
            model,
            optimizer,
            loss_fn,
            device,
            non_blocking,
            prepare_batch,
            model_transform,
            output_transform,
            _scaler,
            gradient_accumulation_steps,
            model_fn,
        )
    elif mode == "apex":
        _update = supervised_training_step_apex(
            model,
            optimizer,
            loss_fn,
            device,
            non_blocking,
            prepare_batch,
            model_transform,
            output_transform,
            gradient_accumulation_steps,
            model_fn,
        )
    elif mode == "tpu":
        _update = supervised_training_step_tpu(
            model,
            optimizer,
            loss_fn,
            device,
            non_blocking,
            prepare_batch,
            model_transform,
            output_transform,
            gradient_accumulation_steps,
            model_fn,
        )
    else:
        _update = supervised_training_step(
            model,
            optimizer,
            loss_fn,
            device,
            non_blocking,
            prepare_batch,
            model_transform,
            output_transform,
            gradient_accumulation_steps,
            model_fn,
        )

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)
    if _scaler and scaler and isinstance(scaler, bool):
        trainer.state.scaler = _scaler  # type: ignore[attr-defined]

    return trainer

