    def save(self,
             is_exit: bool = False,
             force_save_optimizer: bool = False) -> None:
        """ Backup and save the model and state file.

        Parameters
        ----------
        is_exit: bool, optional
            ``True`` if the save request has come from an exit process request otherwise ``False``.
            Default: ``False``
        force_save_optimizer: bool, optional
            ``True`` to force saving the optimizer weights with the model, otherwise ``False``.
            Default:``False``

        Notes
        -----
        The backup function actually backups the model from the previous save iteration rather than
        the current save iteration. This is not a bug, but protection against long save times, as
        models can get quite large, so renaming the current model file rather than copying it can
        save substantial amount of time.
        """
        logger.debug("Backing up and saving models")
        print("")  # Insert a new line to avoid spamming the same row as loss output
        save_averages = self._get_save_averages()
        if save_averages and self._should_backup(save_averages):
            self._backup.backup_model(self.filename)
            self._backup.backup_model(self._plugin.state.filename)

        include_optimizer = (force_save_optimizer or
                             self._save_optimizer == "always" or
                             (self._save_optimizer == "exit" and is_exit))

        try:
            self._plugin.model.save(self.filename, include_optimizer=include_optimizer)
        except ValueError as err:
            if include_optimizer and "name already exists" in str(err):
                logger.warning("Due to a bug in older versions of Tensorflow, optimizer state "
                               "cannot be saved for this model.")
                self._plugin.model.save(self.filename, include_optimizer=False)
            else:
                raise

        self._plugin.state.save()

        msg = "[Saved optimizer state for Snapshot]" if force_save_optimizer else "[Saved model]"
        if save_averages:
            lossmsg = [f"face_{side}: {avg:.5f}"
                       for side, avg in zip(("a", "b"), save_averages)]
            msg += f" - Average loss since last save: {', '.join(lossmsg)}"
        logger.info(msg)