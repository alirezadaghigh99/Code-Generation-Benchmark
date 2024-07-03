def assert_bytes_file_content_correct(file_path: str, content: bytes) -> None:
    with open(file_path, "rb") as f:
        assert f.read() == content