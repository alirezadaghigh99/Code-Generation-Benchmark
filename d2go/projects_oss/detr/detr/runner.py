    def get_default_cfg(cls):
        _C = super().get_default_cfg()
        add_detr_config(_C)
        add_deit_backbone_config(_C)
        add_pit_backbone_config(_C)
        return _C