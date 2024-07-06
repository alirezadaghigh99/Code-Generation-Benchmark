def is_fbcode():
    return not hasattr(torch.version, "git_version")

