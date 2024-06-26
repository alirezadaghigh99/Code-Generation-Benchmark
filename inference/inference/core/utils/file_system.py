def read_text_file(
    path: str,
    split_lines: bool = False,
    strip_white_chars: bool = False,
) -> Union[str, List[str]]:
    with open(path) as f:
        if split_lines:
            lines = list(f.readlines())
            if strip_white_chars:
                return [line.strip() for line in lines if len(line.strip()) > 0]
            else:
                return lines
        content = f.read()
        if strip_white_chars:
            content = content.strip()
        return content