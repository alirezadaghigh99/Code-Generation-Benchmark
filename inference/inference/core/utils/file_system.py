def ensure_parent_dir_exists(path: str) -> None:
    absolute_path = os.path.abspath(path)
    parent_dir = os.path.dirname(absolute_path)
    os.makedirs(parent_dir, exist_ok=True)