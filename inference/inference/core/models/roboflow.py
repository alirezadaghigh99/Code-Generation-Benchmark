def is_model_artefacts_bucket_available() -> bool:
    return (
        AWS_ACCESS_KEY_ID is not None
        and AWS_SECRET_ACCESS_KEY is not None
        and LAMBDA
        and S3_CLIENT is not None
    )

def get_color_mapping_from_environment(
    environment: Optional[dict], class_names: List[str]
) -> Dict[str, str]:
    if color_mapping_available_in_environment(environment=environment):
        return environment["COLORS"]
    return {
        class_name: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
        for i, class_name in enumerate(class_names)
    }

def get_class_names_from_environment_file(environment: Optional[dict]) -> List[str]:
    if environment is None:
        raise ModelArtefactError(
            f"Missing environment while attempting to get model class names."
        )
    if class_mapping_not_available_in_environment(environment=environment):
        raise ModelArtefactError(
            f"Missing `CLASS_MAP` in environment or `CLASS_MAP` is not dict."
        )
    class_names = []
    for i in range(len(environment["CLASS_MAP"].keys())):
        class_names.append(environment["CLASS_MAP"][str(i)])
    return class_names

