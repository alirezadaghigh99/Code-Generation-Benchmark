def build_coco_class_index_mapping(
    coco_categories: List[dict], target_classes: List[str]
) -> Dict[int, int]:
    source_class_to_index = {
        category["name"]: category["id"] for category in coco_categories
    }
    return {
        source_class_to_index[target_class_name]: target_class_index
        for target_class_index, target_class_name in enumerate(target_classes)
    }