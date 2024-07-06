def get_label_indices(num_labels: int, sample_label: str) -> list:
    """
    Function to get sample label indices for a given number
    of labels and a sampling policy
    :param num_labels: int number of labels
    :param sample_label: method for sampling the labels
    :return: list of labels defined by the sampling method.
    """
    if sample_label == "sample":  # sample a random label
        return [random.randrange(num_labels)]
    elif sample_label == "all":  # use all labels
        return list(range(num_labels))
    else:
        raise ValueError("Unknown label sampling policy %s" % sample_label)

def get_h5_sorted_keys(filename: str) -> List[str]:
    """
    Function to get sorted keys from filename
    :param filename: h5 file.
    :return: sorted keys of h5 file.
    """
    with h5py.File(filename, "r") as h5_file:
        return sorted(h5_file.keys())

