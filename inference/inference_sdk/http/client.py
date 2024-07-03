    def use_model(self, model_id: str) -> Generator["InferenceHTTPClient", None, None]:
        previous_model = self.__selected_model
        self.__selected_model = model_id
        try:
            yield self
        finally:
            self.__selected_model = previous_model