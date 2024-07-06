def is_something_to_do(
    output_location: Optional[str],
    display: bool,
    visualise: bool,
) -> bool:
    return output_location is not None or (display is True and visualise is True)

def prepare_target_path(
    reference: Union[str, int],
    output_location: str,
    extension: str,
) -> str:
    if issubclass(type(reference), int):
        reference_number = str(reference).zfill(6)
        file_name = f"frame_{reference_number}.{extension}"
    else:
        file_name = ".".join(os.path.basename(reference).split(".")[:-1])
        file_name = f"{file_name}_prediction.{extension}"
    return os.path.join(output_location, file_name)

