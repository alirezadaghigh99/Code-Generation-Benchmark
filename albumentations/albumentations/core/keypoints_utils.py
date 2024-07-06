def convert_keypoint_to_albumentations(
    keypoint: KeypointType,
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointType:
    if source_format not in keypoint_formats:
        raise ValueError(f"Unknown target_format {source_format}. Supported formats are: {keypoint_formats}")

    if source_format == "xy":
        (x, y), tail = keypoint[:2], tuple(keypoint[2:])
        a, s = 0.0, 0.0
    elif source_format == "yx":
        (y, x), tail = keypoint[:2], tuple(keypoint[2:])
        a, s = 0.0, 0.0
    elif source_format == "xya":
        (x, y, a), tail = keypoint[:3], tuple(keypoint[3:])
        s = 0.0
    elif source_format == "xys":
        (x, y, s), tail = keypoint[:3], tuple(keypoint[3:])
        a = 0.0
    elif source_format == "xyas":
        (x, y, a, s), tail = keypoint[:4], tuple(keypoint[4:])
    elif source_format == "xysa":
        (x, y, s, a), tail = keypoint[:4], tuple(keypoint[4:])
    else:
        raise ValueError(f"Unsupported source format. Got {source_format}")

    if angle_in_degrees:
        a = math.radians(a)

    # if we do not truncate keypoints to integer values, they get into wrong positions after the rotation
    # https://github.com/albumentations-team/albumentations/issues/1819
    keypoint = (int(x), int(y), angle_to_2pi_range(a), s, *tail)
    if check_validity:
        check_keypoint(keypoint, rows, cols)
    return keypoint

def convert_keypoint_from_albumentations(
    keypoint: KeypointType,
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointType:
    if target_format not in keypoint_formats:
        raise ValueError(f"Unknown target_format {target_format}. Supported formats are: {keypoint_formats}")

    (x, y, angle, scale), tail = keypoint[:4], tuple(keypoint[4:])
    angle = angle_to_2pi_range(angle)
    if check_validity:
        check_keypoint((x, y, angle, scale), rows, cols)
    if angle_in_degrees:
        angle = math.degrees(angle)

    if target_format == "xy":
        return (x, y, *tail)
    if target_format == "yx":
        return (y, x, *tail)
    if target_format == "xya":
        return (x, y, angle, *tail)
    if target_format == "xys":
        return (x, y, scale, *tail)
    if target_format == "xyas":
        return (x, y, angle, scale, *tail)
    if target_format == "xysa":
        return (x, y, scale, angle, *tail)

    raise ValueError(f"Invalid target format. Got: {target_format}")

