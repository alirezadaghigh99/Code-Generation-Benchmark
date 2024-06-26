def collect_env() -> dict:
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMCV'] = mmcv.__version__
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMYOLO'] = mmyolo.__version__ + '+' + get_git_hash()[:7]
    return env_info