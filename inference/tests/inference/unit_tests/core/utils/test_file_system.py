def assert_text_file_content_correct(file_path: str, content: str) -> None:
    with open(file_path) as f:
        assert f.read() == content