def _build_teacher(cfg) -> nn.Module:
    """Create teacher using config settings

    Supports torchscript or creating pytorch model using config.
    """
    _validate_teacher_config(cfg)
    if cfg.DISTILLATION.TEACHER.TYPE == "torchscript":
        with PathManager.open(cfg.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME, "rb") as f:
            model = torch.jit.load(f)
    elif cfg.DISTILLATION.TEACHER.TYPE == "config":
        from d2go.runner import import_runner
        from d2go.setup import create_cfg_from_cli

        # teacher config may be set to cuda
        # if user wants to run teacher on cpu only machine by specifying teacher.device,
        # need to override device to cpu before building model
        if cfg.DISTILLATION.TEACHER.DEVICE:
            cfg.DISTILLATION.TEACHER.OVERWRITE_OPTS.extend(
                ["MODEL.DEVICE", cfg.DISTILLATION.TEACHER.DEVICE]
            )

        teacher_cfg = create_cfg_from_cli(
            cfg.DISTILLATION.TEACHER.CONFIG_FNAME,
            cfg.DISTILLATION.TEACHER.OVERWRITE_OPTS,
            cfg.DISTILLATION.TEACHER.RUNNER_NAME,
        )
        runner = import_runner(cfg.DISTILLATION.TEACHER.RUNNER_NAME)()
        model = runner.build_model(teacher_cfg, eval_only=True)
    elif cfg.DISTILLATION.TEACHER.TYPE == "no_teacher":
        model = nn.Identity()
    else:
        raise ValueError(f"Unexpected teacher type: {cfg.DISTILLATION.TEACHER.TYPE}")

    # move teacher to same device as student unless specified
    device = torch.device(cfg.DISTILLATION.TEACHER.DEVICE or cfg.MODEL.DEVICE)
    model = _set_device(model, device)
    model.eval()
    return model