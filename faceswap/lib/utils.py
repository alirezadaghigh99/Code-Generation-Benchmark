    def _get(self) -> None:
        """ Check the model exists, if not, download the model, unzip it and place it in the
        model's cache folder. """
        if self._model_exists:
            self.logger.debug("Model exists: %s", self.model_path)
            return
        self._download_model()
        self._unzip_model()
        os.remove(self._model_zip_path)