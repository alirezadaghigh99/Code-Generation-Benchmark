def leaf_components(input: tf.Module) -> Mapping[Path, LeafVariable]:
    return _get_leaf_components(input)

def multiple_assign(module: tf.Module, parameters: Mapping[Path, tf.Tensor]) -> None:
    """
    Multiple assign takes a dictionary with new values. Dictionary keys are paths to the
    `tf.Variable`s or `gpflow.Parameter` of the input module.

    :param module: `tf.Module`.
    :param parameters: a dictionary with keys of the form ".module.path.to.variable" and new value tensors.
    """
    reference_var_dict = parameter_dict(module)
    for path, value in parameters.items():
        reference_var_dict[path].assign(value)

def freeze(input_module: M) -> M:
    """
    Returns a deepcopy of the input tf.Module with constants instead of variables and parameters.

    :param input_module: tf.Module or gpflow.Module.
    :return: Returns a frozen deepcopy of an input object.
    """
    objects_to_freeze = _get_leaf_components(input_module)
    memo_tensors = {id(v): tf.convert_to_tensor(v) for v in objects_to_freeze.values()}
    module_copy = deepcopy(input_module, memo_tensors)
    return module_copy

def tabulate_module_summary(module: tf.Module, tablefmt: Optional[str] = None) -> str:
    def get_transform(path: Path, var: LeafComponent) -> Optional[str]:
        if hasattr(var, "transform") and var.transform is not None:
            if isinstance(var.transform, tfp.bijectors.Chain):
                return " + ".join(b.__class__.__name__ for b in var.transform.bijectors[::-1])
            return var.transform.__class__.__name__  # type: ignore[no-any-return]
        return None

    def get_prior(path: Path, var: LeafComponent) -> Optional[str]:
        if hasattr(var, "prior") and var.prior is not None:
            return var.prior.name  # type: ignore[no-any-return]
        return None

    # list of (column_name: str, column_getter: Callable[[tf.Variable], str]) tuples:
    column_definition = [
        ("name", lambda path, var: path),
        ("class", lambda path, var: var.__class__.__name__),
        ("transform", get_transform),
        ("prior", get_prior),
        ("trainable", lambda path, var: var.trainable),
        ("shape", lambda path, var: var.shape),
        ("dtype", lambda path, var: var.dtype.name),
        ("value", lambda path, var: _str_tensor_value(var.numpy())),
    ]
    column_names, column_getters = zip(*column_definition)

    merged_leaf_components = _merge_leaf_components(leaf_components(module))

    column_values = [
        [getter(path, variable) for getter in column_getters]
        for path, variable in merged_leaf_components.items()
    ]
    # mypy claims it's wrong to pass a `None` tablefmt below. I think `tabulate` has bad type hints.
    return tabulate(column_values, headers=column_names, tablefmt=tablefmt)  # type: ignore[arg-type]

