def compare_detection_response(
    response, expected_response, type="object_detection", multilabel=False
):
    try:
        assert "time" in response
    except AssertionError:
        raise ValueError(f"Invalid response: {response}, Missing 'time' field.")
    # try:
    #     assert response["time"] == pytest.approx(
    #         expected_response["time"], rel=None, abs=TIME_TOLERANCE
    #     )
    # except AssertionError:
    #     raise ValueError(
    #         f"Invalid response: {response}, 'time' field does not match expected value. Expected {expected_response['time']}, got {response['time']}."
    #     )
    try:
        assert "image" in response
    except AssertionError:
        raise ValueError(f"Invalid response: {response}, Missing 'image' field.")
    try:
        assert response["image"]["width"] == expected_response["image"]["width"]
    except AssertionError:
        raise ValueError(
            f"Invalid response: {response}, 'image' field does not match expected value. Expected {expected_response['image']['width']}, got {response['image']['width']}."
        )
    try:
        assert response["image"]["height"] == expected_response["image"]["height"]
    except AssertionError:
        raise ValueError(
            f"Invalid response: {response}, 'image' field does not match expected value. Expected {expected_response['image']['height']}, got {response['image']['height']}."
        )
    try:
        assert "predictions" in response
    except AssertionError:
        raise ValueError(f"Invalid response: {response}, Missing 'predictions' field.")
    try:
        assert len(response["predictions"]) == len(expected_response["predictions"])
    except AssertionError:
        raise ValueError(
            f"Invalid response: {response}, number of predictions does not match expected value. Expected {len(expected_response['predictions'])} predictions, got {len(response['predictions'])}."
        )
    if type in ["object_detection", "instance_segmentation"]:
        for i, prediction in enumerate(response["predictions"]):
            try:
                assert prediction["x"] == pytest.approx(
                    expected_response["predictions"][i]["x"],
                    rel=None,
                    abs=PIXEL_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'x' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['x']}, got {prediction['x']}."
                )
            try:
                assert prediction["y"] == pytest.approx(
                    expected_response["predictions"][i]["y"],
                    rel=None,
                    abs=PIXEL_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'y' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['y']}, got {prediction['y']}."
                )
            try:
                assert prediction["width"] == pytest.approx(
                    expected_response["predictions"][i]["width"],
                    rel=None,
                    abs=PIXEL_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'width' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['width']}, got {prediction['width']}."
                )
            try:
                assert prediction["height"] == pytest.approx(
                    expected_response["predictions"][i]["height"],
                    rel=None,
                    abs=PIXEL_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'height' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['height']}, got {prediction['height']}."
                )
            try:
                assert prediction["confidence"] == pytest.approx(
                    expected_response["predictions"][i]["confidence"],
                    rel=None,
                    abs=CONFIDENCE_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'confidence' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['confidence']}, got {prediction['confidence']}."
                )
            try:
                assert (
                    prediction["class"] == expected_response["predictions"][i]["class"]
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'class' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['class']}, got {prediction['class']}."
                )
            if type == "instance_segmentation":
                try:
                    assert "points" in prediction
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Missing 'points' field for prediction {i}."
                    )
                for j, point in enumerate(prediction["points"]):
                    try:
                        assert point["x"] == pytest.approx(
                            expected_response["predictions"][i]["points"][j]["x"],
                            rel=None,
                            abs=PIXEL_TOLERANCE,
                        )
                    except AssertionError:
                        raise ValueError(
                            f"Invalid response: {response}, 'x' field does not match expected value for prediction {i}, point {j}. Expected {expected_response['predictions'][i]['points'][j]['x']}, got {point['x']}."
                        )
                    try:
                        assert point["y"] == pytest.approx(
                            expected_response["predictions"][i]["points"][j]["y"],
                            rel=None,
                            abs=PIXEL_TOLERANCE,
                        )
                    except AssertionError:
                        raise ValueError(
                            f"Invalid response: {response}, 'y' field does not match expected value for prediction {i}, point {j}. Expected {expected_response['predictions'][i]['points'][j]['y']}, got {point['y']}."
                        )
    elif type == "classification":
        if multilabel:
            for class_name, confidence in response["predictions"].items():
                try:
                    assert class_name in expected_response["predictions"]
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Unexpected class {class_name}. Expected classes: {expected_response['predictions'].keys()}."
                    )
                try:
                    assert "confidence" in confidence
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Missing 'confidence' field for class {class_name}."
                    )
                try:
                    assert confidence["confidence"] == pytest.approx(
                        expected_response["predictions"][class_name]["confidence"],
                        rel=None,
                        abs=CONFIDENCE_TOLERANCE,
                    )
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, 'confidence' field does not match expected value for class {class_name}. Expected {expected_response['predictions'][class_name]['confidence']}, got {confidence['confidence']}."
                    )
            try:
                assert "predicted_classes" in response
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, Missing 'predicted_classes' field."
                )
            for class_name in response["predicted_classes"]:
                try:
                    assert class_name in expected_response["predictions"]
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Unexpected class {class_name}. Expected classes: {expected_response['predicted_classes']}."
                    )
        else:
            try:
                assert "top" in response
            except AssertionError:
                raise ValueError(f"Invalid response: {response}, Missing 'top' field.")
            try:
                assert response["top"] == expected_response["top"]
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'top' field does not match expected value. Expected {expected_response['top']}, got {response['top']}."
                )
            try:
                assert "confidence" in response
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, Missing 'confidence' field."
                )
            try:
                assert response["confidence"] == pytest.approx(
                    expected_response["confidence"],
                    rel=None,
                    abs=CONFIDENCE_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'confidence' field does not match expected value. Expected {expected_response['confidence']}, got {response['confidence']}."
                )
            for i, prediction in enumerate(response["predictions"]):
                try:
                    assert "class" in prediction
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Missing 'class' field for prediction {i}."
                    )
                try:
                    assert "confidence" in prediction
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Missing 'confidence' field for prediction {i}."
                    )
                try:
                    assert prediction["confidence"] == pytest.approx(
                        expected_response["predictions"][i]["confidence"],
                        rel=None,
                        abs=CONFIDENCE_TOLERANCE,
                    )
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, 'confidence' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['confidence']}, got {prediction['confidence']}."
                    )
                try:
                    assert (
                        prediction["class"]
                        == expected_response["predictions"][i]["class"]
                    )
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, 'class' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['class']}, got {prediction['class']}."
                    )