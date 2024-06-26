def divide_by(array_list: list, num_workers: int):
    """Divide a list of parameters by an integer num_workers.

    :param array_list:
    :param num_workers:
    :return:
    """
    return [x / num_workers for x in array_list]