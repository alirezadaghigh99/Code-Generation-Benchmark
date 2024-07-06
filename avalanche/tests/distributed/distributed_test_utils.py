def check_skip_distributed_test() -> bool:
    return os.environ.get("DISTRIBUTED_TESTS", "false").lower() not in ["1", "true"]

