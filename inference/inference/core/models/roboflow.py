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