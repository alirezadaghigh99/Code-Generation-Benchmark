def apply_args_defaults(
    args: List[Any], kwargs: Dict[str, Any], args_signature=List[Union[str, Tuple[str, Any]]]
) -> Dict[str, Any]:
    """
    Combines positional arguments (`args`) and keyword arguments (`kwargs`)
    according to the provided `args_signature`.

    The `args_signature` is a list that defines the expected arguments.
    Each element in the list can be either:

    - string: This represents the name of an argument expected to be a positional argument.
    - tuple: This represents the name and default value of an argument.
        - The first element in the tuple is the argument name.
        - The second element in the tuple is the default value.

    :param args: List of positional arguments.
    :param kwargs: Dictionary of keyword arguments.
    :param args_signature: List defining the expected arguments as described above.

    :return: A dictionary combining arguments from `args` and `kwargs` according to the `args_signature`.
    """
    # Manual defines function signature necessary because inspection of torch function is not available
    # https://github.com/pytorch/pytorch/issues/74539

    args_dict: Dict[str, Any] = dict()
    for idx, arg_desc in enumerate(args_signature):
        if isinstance(arg_desc, str):
            if arg_desc in kwargs:
                args_dict[arg_desc] = kwargs[arg_desc]
            elif idx < len(args):
                args_dict[arg_desc] = args[idx]
            else:
                raise ValueError("Incorrect args_signature, can not by applied to function arguments.")
        elif isinstance(arg_desc, Tuple):
            arg_name, default = arg_desc
            args_dict[arg_name] = kwargs.get(arg_name, args[idx] if idx < len(args) else default)
        else:
            raise ValueError("Incorrect args_signature, element of list should be str or tuple.")
    return args_dict

