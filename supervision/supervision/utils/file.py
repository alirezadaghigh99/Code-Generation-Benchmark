def read_txt_file(file_path: Union[str, Path], skip_empty: bool = False) -> List[str]:
    """
    Read a text file and return a list of strings without newline characters.
    Optionally skip empty lines.

    Args:
        file_path (Union[str, Path]): The file path as a string or Path object.
        skip_empty (bool): If True, skip lines that are empty or contain only
            whitespace. Default is False.

    Returns:
        List[str]: A list of strings representing the lines in the text file.
    """
    with open(str(file_path), "r") as file:
        if skip_empty:
            lines = [line.rstrip("\n") for line in file if line.strip()]
        else:
            lines = [line.rstrip("\n") for line in file]

    return lines