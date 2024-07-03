def track_progress_rich(func: Callable,
                        tasks: Iterable = tuple(),
                        task_num: int = None,
                        nproc: int = 1,
                        chunksize: int = 1,
                        description: str = 'Processing',
                        color: str = 'blue') -> list:
    """Track the progress of parallel task execution with a progress bar. The
    built-in :mod:`multiprocessing` module is used for process pools and tasks
    are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (Iterable or Sized): A tuple of tasks. There are several cases
            for different format tasks:
            - When ``func`` accepts no arguments: tasks should be an empty
              tuple, and ``task_num`` must be specified.
            - When ``func`` accepts only one argument: tasks should be a tuple
              containing the argument.
            - When ``func`` accepts multiple arguments: tasks should be a
              tuple, with each element representing a set of arguments.
              If an element is a ``dict``, it will be parsed as a set of
              keyword-only arguments.
            Defaults to an empty tuple.
        task_num (int, optional): If ``tasks`` is an iterator which does not
            have length, the number of tasks can be provided by ``task_num``.
            Defaults to None.
        nproc (int): Process (worker) number, if nuproc is 1,
            use single process. Defaults to 1.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
            Defaults to 1.
        description (str): The description of progress bar.
            Defaults to "Process".
        color (str): The color of progress bar. Defaults to "blue".

    Examples:
        >>> import time

        >>> def func(x):
        ...    time.sleep(1)
        ...    return x**2
        >>> track_progress_rich(func, range(10), nproc=2)

    Returns:
        list: The task results.
    """
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    if isinstance(tasks, Sized):
        if len(tasks) == 0:
            if task_num is None:
                raise ValueError('If tasks is an empty iterable, '
                                 'task_num must be set')
            else:
                tasks = tuple(tuple() for _ in range(task_num))
        else:
            if task_num is not None and task_num != len(tasks):
                raise ValueError('task_num does not match the length of tasks')
            task_num = len(tasks)

    if nproc <= 0:
        raise ValueError('nproc must be a positive number')

    skip_times = nproc * chunksize if nproc > 1 else 0
    prog_bar = Progress(
        TextColumn('{task.description}'),
        BarColumn(),
        _SkipFirstTimeRemainingColumn(skip_times=skip_times),
        MofNCompleteColumn(),
        TaskProgressColumn(show_speed=True),
    )

    worker = _Worker(func)
    task_id = prog_bar.add_task(
        total=task_num, color=color, description=description)
    tasks = _tasks_with_index(tasks)

    # Use single process when nproc is 1, else use multiprocess.
    with prog_bar:
        if nproc == 1:
            results = []
            for task in tasks:
                results.append(worker(task)[0])
                prog_bar.update(task_id, advance=1, refresh=True)
        else:
            with Pool(nproc) as pool:
                results = []
                unordered_results = []
                gen = pool.imap_unordered(worker, tasks, chunksize)
                try:
                    for result in gen:
                        result, idx = result
                        unordered_results.append((result, idx))
                        results.append(None)
                        prog_bar.update(task_id, advance=1, refresh=True)
                except Exception as e:
                    prog_bar.stop()
                    raise e
            for result, idx in unordered_results:
                results[idx] = result
    return results