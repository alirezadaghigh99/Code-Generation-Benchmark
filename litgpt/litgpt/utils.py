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
    return optimizer

def extend_checkpoint_dir(checkpoint_dir: Path) -> Path:
    new_checkpoint_dir = "checkpoints" / checkpoint_dir
    should_return_new_dir = (not checkpoint_dir.is_dir() and
                             checkpoint_dir.parts[0] != "checkpoints" and
                             not checkpoint_dir.is_absolute() and
                             new_checkpoint_dir.exists())
    return new_checkpoint_dir if should_return_new_dir else checkpoint_dir

def check_file_size_on_cpu_and_warn(checkpoint_path, device, size_limit=4_509_715_660):
    """
    Checks the file size and raises a warning if it exceeds the size_limit.
    The default size limit is 4.2 GB, the size of TinyLlama 1.1B: 4.2 * 1024 * 1024 * 1024 = 4_509_715_660
    """
    size = 0.0
    if os.path.exists(checkpoint_path):
        size = os.path.getsize(checkpoint_path)
        if size > size_limit and str(device) == "cpu":
            warnings.warn(
                f"The file size of {checkpoint_path} is over {size_limit/1024/1024/1024:.1f} GB. Using a model "
                "with more than 1B parameters on a CPU can be slow, it is recommended to switch to a GPU."
            )
    return size

def save_hyperparameters(function: callable, checkpoint_dir: Path) -> None:
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
    parser.save(config, checkpoint_dir / "hyperparameters.yaml", overwrite=True)

def instantiate_torch_optimizer(optimizer, model_parameters, **kwargs):
    if isinstance(optimizer, str):
        optimizer_cls = getattr(torch.optim, optimizer)
        optimizer = optimizer_cls(model_parameters, **kwargs)
    else:
        optimizer = dict(optimizer)  # copy
        optimizer["init_args"].update(kwargs)
        optimizer = instantiate_class(model_parameters, optimizer)
    return optimizer

def CLI(*args: Any, **kwargs: Any) -> Any:
    from jsonargparse import CLI, set_config_read_mode, set_docstring_parse_options

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    return CLI(*args, **kwargs)

def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_index: int = -100,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude

    # lm_head was chunked (we are fine-tuning)
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

        # chunk cross entropy
        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        non_masked_elems = (targets != ignore_index).sum()
        # See [non_masked_elems div note]
        return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

    # lm_head wasn't chunked, chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    non_masked_elems = (targets != ignore_index).sum()
    # [non_masked_elems div note]:
    #   max(1, non_masked_elems) would be more ergonomic to avoid a division by zero. However that
    #   results in a python int which is then passed back to torch division. By using the
    #   `x.maximum(torch.ones_like(x))` pattern we avoid a cudaStreamSynchronize.
    return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))

