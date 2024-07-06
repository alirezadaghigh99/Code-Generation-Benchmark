def dynamo_skipfiles_allow(exclude_from_skipfiles_pattern: str):
    replaced: bool = False
    try:
        # Temporary wrapping, as preparation for removal of trace_rules.FBCODE_SKIP_DIRS_RE
        # Remove dynamo_skipfiles_allow once trace_rules.FBCODE_SKIP_DIRS removed
        original_FBCODE_SKIP_DIRS_RE = copy.deepcopy(trace_rules.FBCODE_SKIP_DIRS_RE)
        new_FBCODE_SKIP_DIRS = {
            s
            for s in trace_rules.FBCODE_SKIP_DIRS
            if exclude_from_skipfiles_pattern not in s
        }
        trace_rules.FBCODE_SKIP_DIRS_RE = re.compile(
            # pyre-ignore
            f".*({'|'.join(map(re.escape, new_FBCODE_SKIP_DIRS))})"
        )
        replaced = True
    except Exception:
        pass
    yield
    if replaced:
        # pyre-ignore
        trace_rules.FBCODE_SKIP_DIRS_RE = original_FBCODE_SKIP_DIRS_RE

