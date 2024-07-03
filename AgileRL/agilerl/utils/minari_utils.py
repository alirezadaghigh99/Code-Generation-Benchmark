def minari_to_agile_buffer(dataset_id, memory, accelerator=None, remote=False):
    minari_dataset = load_minari_dataset(dataset_id, accelerator, remote)

    for episode in minari_dataset.iterate_episodes():
        for num_steps in range(0, len(episode.rewards)):
            observation = episode.observations[num_steps]
            next_observation = episode.observations[num_steps + 1]
            action = episode.actions[num_steps]
            reward = episode.rewards[num_steps]
            terminal = episode.terminations[num_steps]
            memory.save_to_memory(
                observation, action, reward, next_observation, terminal
            )

    return memory