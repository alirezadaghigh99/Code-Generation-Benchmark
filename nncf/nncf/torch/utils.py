def get_all_modules_by_type(
    model, module_types=None, current_scope=None, ignored_scopes=None, target_scopes=None, memo=None
) -> Dict[Scope, Module]:
    if memo is None:
        memo = set()
    if isinstance(module_types, str):
        module_types = [module_types]
    found = OrderedDict()

    if current_scope is None:
        current_scope = Scope()
        current_scope.push(ScopeElement(model.__class__.__name__))
    for name, module in model.named_children():
        if id(module) in memo:
            continue
        memo.add(id(module))
        child_scope_element = ScopeElement(module.__class__.__name__, name)
        child_scope = current_scope.copy()
        child_scope.push(child_scope_element)

        if matches_any(str(child_scope), ignored_scopes):
            continue

        if target_scopes is None or matches_any(str(child_scope), target_scopes):
            if module_types is None or module_types.count(str(type(module).__name__)) != 0:
                found[child_scope] = module
            sub_found = get_all_modules_by_type(
                module,
                module_types,
                current_scope=child_scope,
                ignored_scopes=ignored_scopes,
                target_scopes=target_scopes,
                memo=memo,
            )
            if sub_found:
                found.update(sub_found)
    return found

def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device on which the first model parameters reside.

    :param model: The PyTorch model.
    :return: The device where the first model parameter reside.
        Default cpu if the model has no parameters.
    """

    try:
        device = next(model.parameters()).device
    except StopIteration:
        # The model had no parameters at all, doesn't matter which device to choose
        device = torch.device("cpu")
    return device

