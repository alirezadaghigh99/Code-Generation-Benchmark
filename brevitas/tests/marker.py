def requires_pt_ge(pt_version: str, system: str = None):
    skip = parse(pt_version) > torch_version
    if system is not None:
        skip = skip and platform.system() == system

    def skip_wrapper(f):
        return pytest.mark.skipif(skip, reason=f'Requires Pytorch >= {pt_version}')(f)

    return skip_wrapper

