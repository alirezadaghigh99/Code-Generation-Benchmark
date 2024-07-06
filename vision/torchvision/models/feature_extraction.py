def get_graph_node_names(
    model: nn.Module,
    tracer_kwargs: Optional[Dict[str, Any]] = None,
    suppress_diff_warning: bool = False,
    concrete_args: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Dev utility to return node names in order of execution. See note on node
    names under :func:`create_feature_extractor`. Useful for seeing which node
    names are available for feature extraction. There are two reasons that
    node names can't easily be read directly from the code for a model:

        1. Not all submodules are traced through. Modules from ``torch.nn`` all
           fall within this category.
        2. Nodes representing the repeated application of the same operation
           or leaf module get a ``_{counter}`` postfix.

    The model is traced twice: once in train mode, and once in eval mode. Both
    sets of node names are returned.

    For more details on the node naming conventions used here, please see the
    :ref:`relevant subheading <about-node-names>` in the
    `documentation <https://pytorch.org/vision/stable/feature_extraction.html>`_.

    Args:
        model (nn.Module): model for which we'd like to print node names
        tracer_kwargs (dict, optional): a dictionary of keyword arguments for
            ``NodePathTracer`` (they are eventually passed onto
            `torch.fx.Tracer <https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer>`_).
            By default, it will be set to wrap and make leaf nodes all torchvision ops:
            {"autowrap_modules": (math, torchvision.ops,),"leaf_modules": _get_leaf_modules_for_ops(),}
            WARNING: In case the user provides tracer_kwargs, above default arguments will be appended to the user
            provided dictionary.
        suppress_diff_warning (bool, optional): whether to suppress a warning
            when there are discrepancies between the train and eval version of
            the graph. Defaults to False.
        concrete_args (Optional[Dict[str, any]]): Concrete arguments that should
            not be treated as Proxies. According to the `Pytorch docs
            <https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.trace>`_,
            this parameter's API may not be guaranteed.

    Returns:
        tuple(list, list): a list of node names from tracing the model in
        train mode, and another from tracing the model in eval mode.

    Examples::

        >>> model = torchvision.models.resnet18()
        >>> train_nodes, eval_nodes = get_graph_node_names(model)
    """
    tracer_kwargs = _set_default_tracer_kwargs(tracer_kwargs)
    is_training = model.training
    train_tracer = NodePathTracer(**tracer_kwargs)
    train_tracer.trace(model.train(), concrete_args=concrete_args)
    eval_tracer = NodePathTracer(**tracer_kwargs)
    eval_tracer.trace(model.eval(), concrete_args=concrete_args)
    train_nodes = list(train_tracer.node_to_qualname.values())
    eval_nodes = list(eval_tracer.node_to_qualname.values())
    if not suppress_diff_warning:
        _warn_graph_differences(train_tracer, eval_tracer)
    # Restore training state
    model.train(is_training)
    return train_nodes, eval_nodes

