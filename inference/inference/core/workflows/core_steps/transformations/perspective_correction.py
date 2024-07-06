def pick_largest_perspective_polygons(
    perspective_polygons_batch: Union[
        List[np.ndarray],
        List[List[np.ndarray]],
        List[List[List[int]]],
        List[List[List[List[int]]]],
    ]
) -> List[np.ndarray]:
    if not isinstance(perspective_polygons_batch, (list, Batch)):
        raise ValueError("Unexpected type of input")
    if not perspective_polygons_batch:
        raise ValueError("Unexpected empty batch")
    largest_perspective_polygons: List[np.ndarray] = []
    for polygons in perspective_polygons_batch:
        if polygons is None:
            continue
        if not isinstance(polygons, list) and not isinstance(polygons, np.ndarray):
            raise ValueError("Unexpected type of batch element")
        if len(polygons) == 0:
            raise ValueError("Unexpected empty batch element")
        if isinstance(polygons, np.ndarray):
            if polygons.shape != (4, 2):
                raise ValueError("Unexpected shape of batch element")
            largest_perspective_polygons.append(polygons)
            continue
        if len(polygons) == 4 and all(
            isinstance(p, list) and len(p) == 2 for p in polygons
        ):
            largest_perspective_polygons.append(np.array(polygons))
            continue
        polygons = [p if isinstance(p, np.ndarray) else np.array(p) for p in polygons]
        polygons = [p for p in polygons if p.shape == (4, 2)]
        if not polygons:
            raise ValueError("No batch element consists of 4 vertices")
        polygons = [np.around(p).astype(np.int32) for p in polygons]
        largest_polygon = max(polygons, key=lambda p: cv.contourArea(p))
        largest_perspective_polygons.append(largest_polygon)
    return largest_perspective_polygons

def correct_detections(
    detections: sv.Detections, perspective_transformer: np.array
) -> sv.Detections:
    corrected_detections: List[sv.Detections] = []
    for i in range(len(detections)):
        # copy
        detection = detections[i]
        mask = np.array(detection.mask)
        if (
            not np.array_equal(mask, np.array(None))
            and len(mask) > 0
            and isinstance(mask[0], np.ndarray)
        ):
            polygon = np.array(sv.mask_to_polygons(mask[0]), dtype=np.float32)
            # https://docs.opencv.org/4.9.0/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
            corrected_polygon: np.ndarray = cv.perspectiveTransform(
                src=polygon, m=perspective_transformer
            ).reshape(-1, 2)
            h, w, *_ = detection.mask[0].shape
            detection.mask = np.array(
                [
                    sv.polygon_to_mask(
                        polygon=np.around(corrected_polygon).astype(np.int32),
                        resolution_wh=(w, h),
                    ).astype(bool)
                ]
            )
            detection.xyxy = np.array(
                [
                    np.around(sv.polygon_to_xyxy(polygon=corrected_polygon)).astype(
                        np.int32
                    )
                ]
            )
        else:
            xmin, ymin, xmax, ymax = np.around(detection[i].xyxy[0]).tolist()
            polygon = np.array(
                [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]],
                dtype=np.float32,
            )
            # https://docs.opencv.org/4.9.0/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
            corrected_polygon: np.ndarray = cv.perspectiveTransform(
                src=polygon, m=perspective_transformer
            ).reshape(-1, 2)
            detection.xyxy = np.array(
                [
                    np.around(sv.polygon_to_xyxy(polygon=corrected_polygon)).astype(
                        np.int32
                    )
                ]
            )
        if KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in detection.data:
            corrected_key_points = cv.perspectiveTransform(
                src=np.array(
                    [detection.data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS][0]],
                    dtype=np.float32,
                ),
                m=perspective_transformer,
            ).reshape(-1, 2)
            detection[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = np.array(
                [np.around(corrected_key_points).astype(np.int32)], dtype="object"
            )
        corrected_detections.append(detection)
    return sv.Detections.merge(corrected_detections)

