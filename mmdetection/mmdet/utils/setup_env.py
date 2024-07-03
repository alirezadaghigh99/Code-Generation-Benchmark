def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmdet into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmdet default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmdet`, and all registries will build modules from mmdet's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmdet.datasets  # noqa: F401,F403
    import mmdet.engine  # noqa: F401,F403
    import mmdet.evaluation  # noqa: F401,F403
    import mmdet.models  # noqa: F401,F403
    import mmdet.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmdet')
        if never_created:
            DefaultScope.get_instance('mmdet', scope_name='mmdet')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmdet':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmdet", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmdet". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmdet-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmdet')