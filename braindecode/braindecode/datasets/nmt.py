def _correct_path(path: str):
    """
    Check if the path is correct and rename the file if needed.

    Parameters
    ----------
    path: basestring
        Path to the file.

    Returns
    -------
    path: basestring
        Corrected path.
    """
    if not Path(path).exists():
        unzip_file_name = f"{NMT_archive_name}.unzip"
        if (Path(path).parent / unzip_file_name).exists():
            try:
                os.rename(
                    src=Path(path).parent / unzip_file_name,
                    dst=Path(path),
                )

            except PermissionError:
                raise PermissionError(
                    f"Please rename {Path(path).parent / unzip_file_name}"
                    + f"manually to {path} and try again."
                )
        path = os.path.join(path, "nmt_scalp_eeg_dataset")

    return path