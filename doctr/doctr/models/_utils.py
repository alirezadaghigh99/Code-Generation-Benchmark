def invert_data_structure(
    x: Union[List[Dict[str, Any]], Dict[str, List[Any]]],
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """Invert a List of Dict of elements to a Dict of list of elements and the other way around

    Args:
    ----
        x: a list of dictionaries with the same keys or a dictionary of lists of the same length

    Returns:
    -------
        dictionary of list when x is a list of dictionaries or a list of dictionaries when x is dictionary of lists
    """
    if isinstance(x, dict):
        assert len({len(v) for v in x.values()}) == 1, "All the lists in the dictionnary should have the same length."
        return [dict(zip(x, t)) for t in zip(*x.values())]
    elif isinstance(x, list):
        return {k: [dic[k] for dic in x] for k in x[0]}
    else:
        raise TypeError(f"Expected input to be either a dict or a list, got {type(input)} instead.")