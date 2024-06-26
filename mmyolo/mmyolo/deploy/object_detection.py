    def register_all_modules(cls):
        """register all modules."""
        from mmdet.utils.setup_env import \
            register_all_modules as register_all_modules_mmdet

        from mmyolo.utils.setup_env import \
            register_all_modules as register_all_modules_mmyolo

        cls.register_deploy_modules()
        register_all_modules_mmyolo(True)
        register_all_modules_mmdet(False)