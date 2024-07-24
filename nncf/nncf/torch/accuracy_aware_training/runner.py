class PTAccuracyAwareTrainingRunner(BaseAccuracyAwareTrainingRunner):
    """
    BaseAccuracyAwareTrainingRunner
    The Training Runner implementation for PyTorch training code.
    """

    def initialize_logging(self, log_dir=None, tensorboard_writer=None):
        if not is_main_process():
            return
        # Only the main process should initialize and create a log directory, other processes don't use it
        super().initialize_logging(log_dir, tensorboard_writer)
        if self._tensorboard_writer is None and TENSORBOARD_AVAILABLE:
            self._tensorboard_writer = SummaryWriter(self._log_dir)

    def validate(self, model):
        with torch.no_grad():
            self.current_val_metric_value = self._validate_fn(model, epoch=self.cumulative_epoch_count)
        is_best = (not self.is_higher_metric_better) != (self.current_val_metric_value > self.best_val_metric_value)
        if is_best:
            self.best_val_metric_value = self.current_val_metric_value
        return self.current_val_metric_value

    def dump_statistics(self, model, compression_controller):
        if not is_main_process():
            return
        super().dump_statistics(model, compression_controller)

    def update_learning_rate(self):
        if self._update_learning_rate_fn is not None:
            self._update_learning_rate_fn(
                self.lr_scheduler, self.training_epoch_count, self.current_val_metric_value, self.current_loss
            )
        else:
            if self.lr_scheduler is not None and self.lr_updates_needed:
                self.lr_scheduler.step(
                    self.training_epoch_count
                    if not isinstance(self.lr_scheduler, ReduceLROnPlateau)
                    else self.best_val_metric_value
                )

    def reset_training(self):
        self.configure_optimizers()

        optimizers = self.optimizer if isinstance(self.optimizer, (tuple, list)) else [self.optimizer]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self.base_lr_reduction_factor_during_search

        lr_schedulers = self.lr_scheduler if isinstance(self.lr_scheduler, (tuple, list)) else [self.lr_scheduler]
        for lr_scheduler in lr_schedulers:
            if lr_scheduler is None:
                continue
            for attr_name in ["base_lrs", "init_lr"]:
                if hasattr(lr_scheduler, attr_name):
                    setattr(
                        lr_scheduler,
                        attr_name,
                        [
                            base_lr * self.base_lr_reduction_factor_during_search
                            for base_lr in getattr(lr_scheduler, attr_name)
                        ],
                    )

        self.training_epoch_count = 0
        self.best_val_metric_value = 0
        self.current_val_metric_value = 0

    def dump_checkpoint(self, model, compression_controller):
        if not is_main_process():
            return
        super().dump_checkpoint(model, compression_controller)

    def load_best_checkpoint(self, model):
        if not is_main_process():
            return
        super().load_best_checkpoint(model)

    def _save_checkpoint(self, model, compression_controller, checkpoint_path):
        optimizers = self.optimizer if isinstance(self.optimizer, (tuple, list)) else [self.optimizer]
        checkpoint = {
            "epoch": self.cumulative_epoch_count + 1,
            "state_dict": model.state_dict(),
            "compression_state": compression_controller.get_compression_state(),
            "best_metric_val": self.best_val_metric_value,
            "current_val_metric_value": self.current_val_metric_value,
            "optimizer": [optimizer.state_dict() for optimizer in optimizers],
        }
        torch.save(checkpoint, checkpoint_path)

    def _load_checkpoint(self, model, checkpoint_path):
        if self._load_checkpoint_fn is not None:
            self._load_checkpoint_fn(model, checkpoint_path)
        else:
            resuming_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            resuming_model_state_dict = resuming_checkpoint.get("state_dict", resuming_checkpoint)
            load_state(model, resuming_model_state_dict, is_resume=True)

    def _make_checkpoint_path(self, is_best, compression_rate=None):
        extension = ".pth"
        return osp.join(self._checkpoint_save_dir, f'acc_aware_checkpoint_{"best" if is_best else "last"}{extension}')

    def add_tensorboard_scalar(self, key, data, step):
        if is_main_process() and self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_scalar(key, data, step)

    def add_tensorboard_image(self, key, data, step):
        if is_main_process() and self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_image(key, ToTensor()(data), step)

