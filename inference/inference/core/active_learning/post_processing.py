def adjust_prediction_to_client_scaling_factor(
    prediction: dict, scaling_factor: float, prediction_type: PredictionType
) -> dict:
    if abs(scaling_factor - 1.0) < 1e-5:
        return prediction
    if "image" in prediction:
        prediction["image"] = {
            "width": round(prediction["image"]["width"] / scaling_factor),
            "height": round(prediction["image"]["height"] / scaling_factor),
        }
    if predictions_should_not_be_post_processed(
        prediction=prediction, prediction_type=prediction_type
    ):
        return prediction
    if prediction_type == INSTANCE_SEGMENTATION_TASK:
        prediction["predictions"] = (
            adjust_prediction_with_bbox_and_points_to_client_scaling_factor(
                predictions=prediction["predictions"],
                scaling_factor=scaling_factor,
                points_key="points",
            )
        )
    if prediction_type == OBJECT_DETECTION_TASK:
        prediction["predictions"] = (
            adjust_object_detection_predictions_to_client_scaling_factor(
                predictions=prediction["predictions"],
                scaling_factor=scaling_factor,
            )
        )
    return prediction