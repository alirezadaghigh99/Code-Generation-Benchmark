def _find_optimizers_for_module(
    module: torch.nn.Module, optimizers: Dict[str, torch.optim.Optimizer]
) -> List[Tuple[str, torch.optim.Optimizer]]:
    """
    Given a module, returns a list of optimizers that are associated with it.
    """
    optimizer_list = []
    module_params = [param.data_ptr() for param in module.parameters()]
    for optim_name, optimizer in optimizers.items():
        optimizer_params = [
            param.data_ptr() for param in optimizer.param_groups[0]["params"]
        ]
        if all(module_param in optimizer_params for module_param in module_params):
            optimizer_list.append((optim_name, optimizer))
    return optimizer_list

