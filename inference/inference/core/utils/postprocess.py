def stretch_keypoints(
    keypoints: np.ndarray,
    infer_shape: Tuple[int, int],
    origin_shape: Tuple[int, int],
) -> np.ndarray:
    scale_width = origin_shape[1] / infer_shape[1]
    scale_height = origin_shape[0] / infer_shape[0]
    for keypoint_id in range(keypoints.shape[1] // 3):
        keypoints[:, keypoint_id * 3] *= scale_width
        keypoints[:, keypoint_id * 3 + 1] *= scale_height
    return keypoints