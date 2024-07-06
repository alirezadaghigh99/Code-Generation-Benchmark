def get_actual_reference_for_current_openvino(rel_path: Path) -> Path:
    """
    Get path to actual reference file.
    If from all of the OpenVINO versions such rel_path is not existed,
    than the path for current OpenVINO version is returned.

    :param rel_path: Relative path to reference file.

    :return: Path to a reference file.
    """
    root_dir = OPENVINO_NATIVE_TEST_ROOT / "data"
    current_ov_version = get_openvino_version()

    def is_valid_version(dir_path: Path) -> bool:
        try:
            version.parse(dir_path.name)
        except version.InvalidVersion:
            return False
        return True

    ref_versions = filter(is_valid_version, root_dir.iterdir())
    ref_versions = sorted(ref_versions, key=lambda x: version.parse(x.name), reverse=True)
    ref_versions = filter(lambda x: version.parse(x.name) <= version.parse(current_ov_version), ref_versions)

    for root_version in ref_versions:
        file_name = root_version / rel_path
        if file_name.is_file():
            return file_name
    return root_dir / current_ov_version / rel_path

