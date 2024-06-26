    def remove(self, model_id: str) -> None:
        """Removes a model from the manager.

        Args:
            model_id (str): The identifier of the model.
        """
        try:
            logger.debug(f"Removing model {model_id} from base model manager")
            self.check_for_model(model_id)
            self._models[model_id].clear_cache()
            del self._models[model_id]
        except InferenceModelNotFound:
            logger.warning(
                f"Attempted to remove model with id {model_id}, but it is not loaded. Skipping..."
            )