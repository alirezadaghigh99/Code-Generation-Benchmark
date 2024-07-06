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

