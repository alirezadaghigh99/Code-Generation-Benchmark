def load_nifti_file(file_path: str) -> np.ndarray:
    """
    :param file_path: path of a Nifti file with suffix .nii or .nii.gz
    :return: return the numpy array
    """
    if not (file_path.endswith(".nii") or file_path.endswith(".nii.gz")):
        raise ValueError(
            f"Nifti file path must end with .nii or .nii.gz, got {file_path}."
        )
    return np.asarray(nib.load(file_path).dataobj, dtype=np.float32)

