def track_tasks(input_path, worker_map=None):
    """
    Takes a path to a folder containing the worker annotation metadata from AWS Sagemaker labeling job and a
    dictionary mapping AWS worker subs to their names or identification tags and returns a dictionary mapping
    the names/identification tags to the number of labeling tasks completed.

    If no worker map is provided, this function returns a dictionary mapping the worker "sub" fields to
    the number of tasks they completed.

    :param input_path: string of the path to the directory containing the worker annotation sub-directories
    :param worker_map: dictionary mapping AWS worker subs to the worker identifications
    :return: dictionary mapping worker identifications to the number of tasks completed
    """
    tracker = {}
    res = {}
    for direc in os.listdir(input_path):
        subdir_path = os.path.join(input_path, direc)
        subdir = os.listdir(subdir_path)
        json_file_path = os.path.join(subdir_path, subdir[0])
        with open(json_file_path) as json_file:
            json_string = json_file.read()
        subs = get_worker_subs(json_string)
        for sub in subs:
            tracker[sub] = tracker.get(sub, 0) + 1

    if worker_map:
        for sub in tracker:
            worker = worker_map[sub]
            res[worker] = tracker[sub]
        return res
    return tracker

