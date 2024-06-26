def is_prediction_a_stub(prediction: Prediction) -> bool:
    return prediction.get("is_stub", False)