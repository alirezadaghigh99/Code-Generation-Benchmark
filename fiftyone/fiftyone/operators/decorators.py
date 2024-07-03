def dir_state(dirpath):
    try:
        state = hash(os.path.getmtime(dirpath))
    except:
        return None

    for p in _iter_plugin_metadata_files(root_dir=dirpath):
        state ^= hash(os.path.getmtime(os.path.dirname(p)))

    return state