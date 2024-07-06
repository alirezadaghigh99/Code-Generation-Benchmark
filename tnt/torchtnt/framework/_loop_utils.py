def _set_module_training_mode(
    modules: Dict[str, nn.Module], mode: bool
) -> Dict[str, bool]:
    """Returns states to allow for a reset at the end of the loop."""
    prior_module_train_states = {}
    for name, module in modules.items():
        prior_module_train_states[name] = module.training
        is_ddp = isinstance(module, DistributedDataParallel)

        if _EXPORT_UTILS_AVAIL and model_is_exported(
            module.module if is_ddp else module
        ):
            if mode:
                module = torch.ao.quantization.move_exported_model_to_train(
                    module.module if is_ddp else module
                )
            else:
                module = torch.ao.quantization.move_exported_model_to_eval(
                    module.module if is_ddp else module
                )
        else:
            module.train(mode)

    return prior_module_train_states

