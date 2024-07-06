def build_anchor_generator(cfg, default_args=None):
    warnings.warn(
        '``build_anchor_generator`` would be deprecated soon, please use '
        '``mmdet.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)

