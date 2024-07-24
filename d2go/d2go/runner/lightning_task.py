def register(self, cfg: CfgNode):
        inject_coco_datasets(cfg)
        register_dynamic_datasets(cfg)
        update_cfg_if_using_adhoc_dataset(cfg)

class GeneralizedRCNNTask(DefaultTask):
    @classmethod
    def get_default_cfg(cls):
        return GeneralizedRCNNRunner.get_default_cfg()

