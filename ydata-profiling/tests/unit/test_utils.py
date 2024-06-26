def patch_arg(d, new_name):
    """Patch until this PR is released: https://github.com/dylan-profiler/visions/pull/172"""
    if isinstance(d["argnames"], str):
        d["argnames"] = d["argnames"].split(",")

    d["argnames"] = [x if x != "type" else new_name for x in d["argnames"]]
    return d