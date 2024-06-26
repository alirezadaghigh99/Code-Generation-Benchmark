def register_datapoint_at_roboflow(cache, strategy_with_spare_credit, encoded_image, local_image_id, prediction, prediction_type, configuration, api_key, batch_name, inference_id=None):
    # Collect tags based on the configuration and strategy
    tags = collect_tags(configuration, strategy_with_spare_credit)

    # Attempt to register the image in Roboflow
    roboflow_image_id = safe_register_image_at_roboflow(
        cache=cache,
        strategy=strategy_with_spare_credit,
        encoded_image=encoded_image,
        local_image_id=local_image_id,
        configuration=configuration,
        api_key=api_key,
        batch_name=batch_name,
        tags=tags,
        inference_id=inference_id
    )

    # Check if prediction registration is prohibited
    if not roboflow_image_id:
        return None

    # Encode the prediction and prediction type
    encoded_prediction, prediction_file_type = encode_prediction(prediction, prediction_type)

    # Annotate the image at Roboflow
    annotate_image_at_roboflow(
        api_key=api_key,
        dataset_id=configuration.dataset_id,
        local_image_id=local_image_id,
        roboflow_image_id=roboflow_image_id,
        encoded_prediction=encoded_prediction,
        prediction_file_type=prediction_file_type,
        is_prediction=True
    )