    def from_cfg(cls, cfg: ConfigType) -> 'FlexibleRunner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg.get('work_dir', 'work_dirs'),
            experiment_name=cfg.get('experiment_name'),
            train_dataloader=cfg.get('train_dataloader'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            train_cfg=cfg.get('train_cfg'),
            val_dataloader=cfg.get('val_dataloader'),
            val_evaluator=cfg.get('val_evaluator'),
            val_cfg=cfg.get('val_cfg'),
            test_dataloader=cfg.get('test_dataloader'),
            test_evaluator=cfg.get('test_evaluator'),
            test_cfg=cfg.get('test_cfg'),
            strategy=cfg.get('strategy'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            cfg=cfg,
        )

        return runner