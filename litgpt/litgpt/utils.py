def instantiate_bnb_optimizer(optimizer, model_parameters):
    if (isinstance(optimizer, str) and "AdamW" not in optimizer) or (isinstance(optimizer, dict) and "AdamW" not in optimizer.get("class_path", "")):
        raise ValueError("The chosen quantization format only supports the AdamW optimizer.")

    import bitsandbytes as bnb
    if isinstance(optimizer, str):
        optimizer = bnb.optim.PagedAdamW(model_parameters)
    else:
        optim_args = get_argument_names(bnb.optim.PagedAdamW)
        allowed_kwargs = {key: optimizer["init_args"][key] for key in optim_args & optimizer["init_args"].keys()}
        optimizer = bnb.optim.PagedAdamW(model_parameters, **allowed_kwargs)
    return optimizerdef extend_checkpoint_dir(checkpoint_dir: Path) -> Path:
    new_checkpoint_dir = "checkpoints" / checkpoint_dir
    should_return_new_dir = (not checkpoint_dir.is_dir() and
                             checkpoint_dir.parts[0] != "checkpoints" and
                             not checkpoint_dir.is_absolute() and
                             new_checkpoint_dir.exists())
    return new_checkpoint_dir if should_return_new_dir else checkpoint_dirdef save_hyperparameters(function: callable, checkpoint_dir: Path) -> None:
    """Captures the CLI parameters passed to `function` without running `function` and saves them to the checkpoint."""
    from jsonargparse import capture_parser

    # TODO: Make this more robust
    # This hack strips away the subcommands from the top-level CLI
    # to parse the file as if it was called as a script
    known_commands = [
        ("finetune_full",),  # For subcommands, use `("finetune", "full")` etc
        ("finetune_lora",),
        ("finetune_adapter",),
        ("finetune_adapter_v2",),
        ("finetune",),
        ("pretrain",),
    ]
    for known_command in known_commands:
        unwanted = slice(1, 1 + len(known_command))
        if tuple(sys.argv[unwanted]) == known_command:
            sys.argv[unwanted] = []

    parser = capture_parser(lambda: CLI(function))
    config = parser.parse_args()
    parser.save(config, checkpoint_dir / "hyperparameters.yaml", overwrite=True)def instantiate_torch_optimizer(optimizer, model_parameters, **kwargs):
    if isinstance(optimizer, str):
        optimizer_cls = getattr(torch.optim, optimizer)
        optimizer = optimizer_cls(model_parameters, **kwargs)
    else:
        optimizer = dict(optimizer)  # copy
        optimizer["init_args"].update(kwargs)
        optimizer = instantiate_class(model_parameters, optimizer)
    return optimizer