def run(args, env_fn, policy_fn, get_weights_fn, set_weights_fn):
    initialize_logger(
        logging_level=logging.getLevelName(args.logging_level))

    if args.n_env > 1:
        args.n_explorer = 1
    elif args.n_explorer is None:
        args.n_explorer = multiprocessing.cpu_count() - 1
    assert args.n_explorer > 0, "[error] number of explorers must be positive integer"

    env = env_fn()

    global_rb, queues, is_training_done, lock, trained_steps = prepare_experiment(env, args)

    noise = 0.3
    tasks = []

    # Add explorers
    if args.n_env > 1:
        tasks.append(Process(
            target=explorer,
            args=[global_rb, queues[0], trained_steps, is_training_done,
                  lock, env_fn, policy_fn, set_weights_fn, noise,
                  args.n_env, args.n_thread, args.local_buffer_size,
                  args.episode_max_steps, args.gpu_explorer]))
    else:
        for i in range(args.n_explorer):
            tasks.append(Process(
                target=explorer,
                args=[global_rb, queues[i], trained_steps, is_training_done,
                      lock, env_fn, policy_fn, set_weights_fn, noise,
                      args.n_env, args.n_thread, args.local_buffer_size,
                      args.episode_max_steps, args.gpu_explorer]))

    # Add learner
    tasks.append(Process(
        target=learner,
        args=[global_rb, trained_steps, is_training_done,
              lock, env_fn(), policy_fn, get_weights_fn,
              args.n_training, args.param_update_freq,
              args.test_freq, args.gpu_learner, queues]))

    # Add evaluator
    tasks.append(Process(
        target=evaluator,
        args=[is_training_done, env_fn(), policy_fn, set_weights_fn,
              queues[-1], args.gpu_evaluator, args.save_model_interval]))

    for task in tasks:
        task.start()
    for task in tasks:
        task.join()