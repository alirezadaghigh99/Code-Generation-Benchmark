def detections_to_coco_annotations(
    detections: Detections,
    image_id: int,
    annotation_id: int,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> Tuple[List[Dict], int]:
    coco_annotations = []
    for xyxy, mask, _, class_id, _, _ in detections:
        box_width, box_height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        segmentation = []
        iscrowd = 0
        if mask is not None:
            iscrowd = contains_holes(mask=mask) or contains_multiple_segments(mask=mask)

            if iscrowd:
                segmentation = {
                    "counts": mask_to_rle(mask=mask),
                    "size": list(mask.shape[:2]),
                }
            else:
                segmentation = [
                    list(
                        approximate_mask_with_polygons(
                            mask=mask,
                            min_image_area_percentage=min_image_area_percentage,
                            max_image_area_percentage=max_image_area_percentage,
                            approximation_percentage=approximation_percentage,
                        )[0].flatten()
                    )
                ]
        coco_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(class_id),
            "bbox": [xyxy[0], xyxy[1], box_width, box_height],
            "area": box_width * box_height,
            "segmentation": segmentation,
            "iscrowd": iscrowd,
        }
        coco_annotations.append(coco_annotation)
        annotation_id += 1
    return coco_annotations, annotation_id