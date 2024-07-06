def set_training(self, is_training: bool) -> None:
        """ Set the internal training flag to the given `is_training` value.

        If a new training session is being instigated, refresh the log filenames

        Parameters
        ----------
        is_training: bool
            ``True`` to indicate that the logs to be read are from the currently training
            session otherwise ``False``
        """
        if self._is_training == is_training:
            logger.debug("Training flag already set to %s. Returning", is_training)
            return

        logger.debug("Setting is_training to %s", is_training)
        self._is_training = is_training
        if is_training:
            self._log_files.refresh()
            log_file = self._log_files.get(self.session_ids[-1])
            logger.debug("Setting training iterator for log file: '%s'", log_file)
            self._training_iterator = tf.compat.v1.io.tf_record_iterator(log_file)
        else:
            logger.debug("Removing training iterator")
            del self._training_iterator
            self._training_iterator = None

