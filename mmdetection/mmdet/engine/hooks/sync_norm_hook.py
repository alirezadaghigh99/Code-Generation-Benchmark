class SyncNormHook(Hook):
    """Synchronize Norm states before validation, currently used in YOLOX."""

    def before_val_epoch(self, runner):
        """Synchronizing norm."""
        module = runner.model
        _, world_size = get_dist_info()
        if world_size == 1:
            return
        norm_states = get_norm_states(module)
        if len(norm_states) == 0:
            return
        # TODO: use `all_reduce_dict` in mmengine
        norm_states = all_reduce_dict(norm_states, op='mean')
        module.load_state_dict(norm_states, strict=False)