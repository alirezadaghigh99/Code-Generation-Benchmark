class CfgNode(_CfgNode):
    @classmethod
    def cast_from_other_class(cls, other_cfg):
        """Cast an instance of other CfgNode to D2Go's CfgNode (or its subclass)"""
        new_cfg = cls(other_cfg)
        # copy all fields inside __dict__, this will preserve fields like __deprecated_keys__
        for k, v in other_cfg.__dict__.items():
            new_cfg.__dict__[k] = v
        return new_cfg

    def merge_from_file(self, cfg_filename: str, *args, **kwargs):
        cfg_filename = reroute_config_path(cfg_filename)
        with reroute_load_yaml_with_base():
            res = super().merge_from_file(cfg_filename, *args, **kwargs)
            self._run_custom_processing(is_dump=False)
            return res

    def merge_from_list(self, cfg_list: List[str]):
        # NOTE: YACS's orignal merge_from_list could not handle non-existed keys even if
        # new_allow is set, override the method for support this.
        override_cfg = _opts_to_dict(cfg_list)
        res = super().merge_from_other_cfg(CfgNode(override_cfg))
        self._run_custom_processing(is_dump=False)
        return res

    def dump(self, *args, **kwargs):
        cfg = copy.deepcopy(self)
        cfg._run_custom_processing(is_dump=True)
        return super(CfgNode, cfg).dump(*args, **kwargs)

    @staticmethod
    def load_yaml_with_base(filename: str, *args, **kwargs):
        filename = reroute_config_path(filename)
        with reroute_load_yaml_with_base():
            return _CfgNode.load_yaml_with_base(filename, *args, **kwargs)

    def __hash__(self):
        # dump follows alphabetical order, thus good for hash use
        return hash(self.dump())

    def _run_custom_processing(self, is_dump=False):
        """Apply config load post custom processing from registry"""
        frozen = self.is_frozen()
        self.defrost()
        for name, process_func in CONFIG_CUSTOM_PARSE_REGISTRY:
            logger.info(f"Apply config processing: {name}, is_dump={is_dump}")
            process_func(self, is_dump)
        if frozen:
            self.freeze()

    def get_default_cfg(self):
        """Return the defaults for this instance of CfgNode"""
        return resolve_default_config(self)