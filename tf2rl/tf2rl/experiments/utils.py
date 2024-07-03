def restore_latest_n_traj(dirname, n_path=10, max_steps=None):
    assert os.path.isdir(dirname)
    filenames = get_filenames(dirname, n_path)
    return load_trajectories(filenames, max_steps)