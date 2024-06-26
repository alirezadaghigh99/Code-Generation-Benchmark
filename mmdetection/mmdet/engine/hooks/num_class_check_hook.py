    def before_val_epoch(self, runner: Runner) -> None:
        """Check whether the dataset in val epoch is compatible with head.

        Args:
            runner (:obj:`Runner`): The runner of the training or evaluation
                process.
        """
        self._check_head(runner, 'val')