def is_prediction_registration_forbidden(
    prediction: Prediction,
    persist_predictions: bool,
    roboflow_image_id: Optional[str],
) -> bool:
    return (
        roboflow_image_id is None
        or persist_predictions is False
        or prediction.get("is_stub", False) is True
        or (len(prediction.get("predictions", [])) == 0 and "top" not in prediction)
    )