    def register(
        self, prediction: InferenceResponse, model_id: str, request: InferenceRequest
    ) -> None:
        try:
            resolved_model_id = resolve_roboflow_model_alias(model_id=model_id)
            target_dataset = (
                request.active_learning_target_dataset
                or resolved_model_id.split("/")[0]
            )
            middleware_key = f"{model_id}->{target_dataset}"
            self.ensure_middleware_initialised(
                model_id=resolved_model_id,
                request=request,
                middleware_key=middleware_key,
                target_dataset=target_dataset,
            )
            self.register_datapoint(
                prediction=prediction,
                model_id=resolved_model_id,
                request=request,
                middleware_key=middleware_key,
            )
        except Exception as error:
            # Error handling to be decided
            logger.warning(
                f"Error in datapoint registration for Active Learning. Details: {error}. "
                f"Error is suppressed in favour of normal operations of API."
            )