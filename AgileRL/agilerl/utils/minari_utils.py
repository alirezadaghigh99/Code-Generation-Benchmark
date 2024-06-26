def minari_to_agile_dataset(dataset_id, remote=False):
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []

    minari_dataset = load_minari_dataset(dataset_id, remote)

    for episode in minari_dataset.iterate_episodes():
        observations.extend(episode.observations[:-1])
        next_observations.extend(episode.observations[1:])
        actions.extend(episode.actions[:])
        rewards.extend(episode.rewards[:])
        terminals.extend(episode.terminations[:])

    agile_dataset_id = dataset_id.split("-")
    agile_dataset_id[0] = agile_dataset_id[0] + "_agile"
    agile_dataset_id = "-".join(agile_dataset_id)

    agile_file_path = get_dataset_path(agile_dataset_id)

    agile_dataset_path = os.path.join(agile_file_path, "data")
    os.makedirs(agile_dataset_path, exist_ok=True)
    data_path = os.path.join(agile_dataset_path, "main_data.hdf5")

    # with h5py.File(os.path.join(agile_file_path, "data", "main_data.hdf5"), "w") as f:
    f = h5py.File(data_path, "w")

    f.create_dataset("observations", data=observations)
    f.create_dataset("next_observations", data=next_observations)
    f.create_dataset("actions", data=actions)
    f.create_dataset("rewards", data=rewards)
    f.create_dataset("terminals", data=terminals)

    return f