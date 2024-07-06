def check_objects_presence_in_consensus_detections(
    consensus_detections: sv.Detections,
    class_aware: bool,
    aggregation_mode: AggregationMode,
    required_objects: Optional[Union[int, Dict[str, int]]],
) -> Tuple[bool, Dict[str, float]]:
    if not consensus_detections:
        return False, {}
    if required_objects is None:
        required_objects = 0
    if isinstance(required_objects, dict) and not class_aware:
        required_objects = sum(required_objects.values())
    if (
        isinstance(required_objects, int)
        and len(consensus_detections) < required_objects
    ):
        return False, {}
    if not class_aware:
        aggregated_confidence = aggregate_field_values(
            detections=consensus_detections,
            field="confidence",
            aggregation_mode=aggregation_mode,
        )
        return True, {"any_object": aggregated_confidence}
    class2detections = {}
    for class_name in set(consensus_detections["class_name"]):
        class2detections[class_name] = consensus_detections[
            consensus_detections["class_name"] == class_name
        ]
    if isinstance(required_objects, dict):
        for requested_class, required_objects_count in required_objects.items():
            if (
                requested_class not in class2detections
                or len(class2detections[requested_class]) < required_objects_count
            ):
                return False, {}
    class2confidence = {
        class_name: aggregate_field_values(
            detections=class_detections,
            field="confidence",
            aggregation_mode=aggregation_mode,
        )
        for class_name, class_detections in class2detections.items()
    }
    return True, class2confidence

def get_consensus_for_single_detection(
    detection: sv.Detections,
    source_id: int,
    detections_from_sources: List[sv.Detections],
    iou_threshold: float,
    class_aware: bool,
    required_votes: int,
    confidence: float,
    detections_merge_confidence_aggregation: AggregationMode,
    detections_merge_coordinates_aggregation: AggregationMode,
    detections_already_considered: Set[str],
) -> Tuple[List[sv.Detections], Set[str]]:
    if detection and detection["detection_id"][0] in detections_already_considered:
        return [], detections_already_considered
    consensus_detections = []
    detections_with_max_overlap = (
        get_detections_from_different_sources_with_max_overlap(
            detection=detection,
            source=source_id,
            detections_from_sources=detections_from_sources,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
            detections_already_considered=detections_already_considered,
        )
    )

    if len(detections_with_max_overlap) < (required_votes - 1):
        # Returning empty sv.Detections
        return consensus_detections, detections_already_considered
    detections_to_merge = sv.Detections.merge(
        [detection]
        + [matched_value[0] for matched_value in detections_with_max_overlap.values()]
    )
    merged_detection = merge_detections(
        detections=detections_to_merge,
        confidence_aggregation_mode=detections_merge_confidence_aggregation,
        boxes_aggregation_mode=detections_merge_coordinates_aggregation,
    )
    if merged_detection.confidence[0] < confidence:
        # Returning empty sv.Detections
        return consensus_detections, detections_already_considered
    consensus_detections.append(merged_detection)
    detections_already_considered.add(detection[DETECTION_ID_KEY][0])
    for matched_value in detections_with_max_overlap.values():
        detections_already_considered.add(matched_value[0][DETECTION_ID_KEY][0])
    return consensus_detections, detections_already_considered

def get_detections_from_different_sources_with_max_overlap(
    detection: sv.Detections,
    source: int,
    detections_from_sources: List[sv.Detections],
    iou_threshold: float,
    class_aware: bool,
    detections_already_considered: Set[str],
) -> Dict[int, Tuple[sv.Detections, float]]:
    current_max_overlap = {}
    for other_source, other_detection in enumerate_detections(
        detections_from_sources=detections_from_sources,
        excluded_source_id=source,
    ):
        if other_detection[DETECTION_ID_KEY][0] in detections_already_considered:
            continue
        if (
            class_aware
            and detection["class_name"][0] != other_detection["class_name"][0]
        ):
            continue
        iou_value = calculate_iou(
            detection_a=detection,
            detection_b=other_detection,
        )
        if iou_value <= iou_threshold:
            continue
        if current_max_overlap.get(other_source) is None:
            current_max_overlap[other_source] = (other_detection, iou_value)
        if current_max_overlap[other_source][1] < iou_value:
            current_max_overlap[other_source] = (other_detection, iou_value)
    return current_max_overlap

def aggregate_field_values(
    detections: sv.Detections,
    field: str,
    aggregation_mode: AggregationMode = AggregationMode.AVERAGE,
) -> float:
    values = []
    if hasattr(detections, field):
        values = getattr(detections, field)
        if isinstance(values, np.ndarray):
            values = values.astype(float).tolist()
    elif hasattr(detections, "data") and field in detections.data:
        values = detections[field]
        if isinstance(values, np.ndarray):
            values = values.astype(float).tolist()
    return AGGREGATION_MODE2FIELD_AGGREGATOR[aggregation_mode](values)

def calculate_iou(detection_a: sv.Detections, detection_b: sv.Detections) -> float:
    iou = float(sv.box_iou_batch(detection_a.xyxy, detection_b.xyxy)[0][0])
    if math.isnan(iou):
        iou = 0
    return iou

def enumerate_detections(
    detections_from_sources: List[sv.Detections],
    excluded_source_id: Optional[int] = None,
) -> Generator[Tuple[int, sv.Detections], None, None]:
    for source_id, detections in enumerate(detections_from_sources):
        if excluded_source_id == source_id:
            continue
        for i in range(len(detections)):
            yield source_id, detections[i]

def agree_on_consensus_for_all_detections_sources(
    detections_from_sources: List[sv.Detections],
    required_votes: int,
    class_aware: bool,
    iou_threshold: float,
    confidence: float,
    classes_to_consider: Optional[List[str]],
    required_objects: Optional[Union[int, Dict[str, int]]],
    presence_confidence_aggregation: AggregationMode,
    detections_merge_confidence_aggregation: AggregationMode,
    detections_merge_coordinates_aggregation: AggregationMode,
) -> Tuple[str, bool, Dict[str, float], sv.Detections]:
    if does_not_detect_objects_in_any_source(
        detections_from_sources=detections_from_sources
    ):
        return "undefined", False, {}, []
    parent_id = get_parent_id_of_detections_from_sources(
        detections_from_sources=detections_from_sources,
    )
    detections_from_sources = filter_predictions(
        predictions=detections_from_sources,
        classes_to_consider=classes_to_consider,
    )
    detections_already_considered = set()
    consensus_detections = []
    for source_id, detection in enumerate_detections(
        detections_from_sources=detections_from_sources
    ):
        (
            consensus_detections_update,
            detections_already_considered,
        ) = get_consensus_for_single_detection(
            detection=detection,
            source_id=source_id,
            detections_from_sources=detections_from_sources,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
            required_votes=required_votes,
            confidence=confidence,
            detections_merge_confidence_aggregation=detections_merge_confidence_aggregation,
            detections_merge_coordinates_aggregation=detections_merge_coordinates_aggregation,
            detections_already_considered=detections_already_considered,
        )
        consensus_detections += consensus_detections_update
    consensus_detections = sv.Detections.merge(consensus_detections)
    (
        object_present,
        presence_confidence,
    ) = check_objects_presence_in_consensus_detections(
        consensus_detections=consensus_detections,
        aggregation_mode=presence_confidence_aggregation,
        class_aware=class_aware,
        required_objects=required_objects,
    )
    return (
        parent_id,
        object_present,
        presence_confidence,
        consensus_detections,
    )

def does_not_detect_objects_in_any_source(
    detections_from_sources: List[sv.Detections],
) -> bool:
    return all(len(p) == 0 for p in detections_from_sources)

def get_largest_bounding_box(detections: sv.Detections) -> Tuple[int, int, int, int]:
    areas: List[float] = detections.area.astype(float).tolist()
    max_area = max(areas)
    max_area_index = areas.index(max_area)
    return tuple(detections[max_area_index].xyxy[0])

def get_smallest_bounding_box(detections: sv.Detections) -> Tuple[int, int, int, int]:
    areas: List[float] = detections.area.astype(float).tolist()
    min_area = min(areas)
    min_area_index = areas.index(min_area)
    return tuple(detections[min_area_index].xyxy[0])

def get_average_bounding_box(detections: sv.Detections) -> Tuple[int, int, int, int]:
    avg_xyxy: np.ndarray = sum(detections.xyxy) / len(detections)
    return tuple(avg_xyxy.astype(float))

def filter_predictions(
    predictions: List[sv.Detections],
    classes_to_consider: Optional[List[str]],
) -> List[sv.Detections]:
    if not classes_to_consider:
        return predictions
    return [
        detections[np.isin(detections["class_name"], classes_to_consider)]
        for detections in predictions
        if "class_name" in detections.data
    ]

def get_parent_id_of_detections_from_sources(
    detections_from_sources: List[sv.Detections],
) -> str:
    encountered_parent_ids = set(
        np.concatenate(
            [
                detections[PARENT_ID_KEY]
                for detections in detections_from_sources
                if PARENT_ID_KEY in detections.data
            ]
        ).tolist()
    )
    if len(encountered_parent_ids) != 1:
        raise ValueError(
            "Missmatch in predictions - while executing consensus step, "
            "in equivalent batches, detections are assigned different parent "
            "identifiers, whereas consensus can only be applied for predictions "
            "made against the same input."
        )
    return next(iter(encountered_parent_ids))

