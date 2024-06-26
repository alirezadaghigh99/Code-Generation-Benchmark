def detect_gazes(frame: np.ndarray):
    img_encode = cv2.imencode(".jpg", frame)[1]
    img_base64 = base64.b64encode(img_encode)
    resp = requests.post(
        GAZE_DETECTION_URL,
        json={
            "api_key": ROBOFLOW_API_KEY,
            "image": {"type": "base64", "value": img_base64.decode("utf-8")},
        },
    )
    # print(resp.json())
    gazes = resp.json()[0]["predictions"]
    return gazes