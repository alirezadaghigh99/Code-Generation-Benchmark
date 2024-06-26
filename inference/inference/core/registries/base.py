    def get_model(self, model_type: str, model_id: str) -> Model:
        """Returns the model class based on the given model type.

        Args:
            model_type (str): The type of the model to be retrieved.
            model_id (str): The ID of the model to be retrieved (unused in the current implementation).

        Returns:
            Model: The model class corresponding to the given model type.

        Raises:
            ModelNotRecognisedError: If the model_type is not found in the registry_dict.
        """
        if model_type not in self.registry_dict:
            raise ModelNotRecognisedError(
                f"Could not find model of type: {model_type} in configured registry."
            )
        return self.registry_dict[model_type]