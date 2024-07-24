def values(cls) -> FrozenSet[str]:
        """Returns all values."""
        return frozenset(arg.value for arg in cls)

def format_param_args(param: ParamName, details: Any) -> Union[ParamArg, List[ParamVal]]:
    """Ensures each user-inputted parameter is a properly typed list.
    Also provides custom support for certain parameters."""
    if not isinstance(details, list):
        details = [details]

    for detail in details:
        if ParamArg.is_arg(detail):
            return ParamArg(detail)

    if param == "layout":
        # if a user inputs layout=[1,2], they wanted "1x2"
        # note that the above conversion to a list of details wouldn't work as expected here
        if all(isinstance(dim, int) for dim in details):
            return ["x".join(map(str, details))]
        # will turn [(1,2), [3,4], "5x6"] into ["1x2", "3x4", "5x6"]
        for i, detail in enumerate(details):
            if isinstance(detail, Iterable) and all(isinstance(dim, int) for dim in detail):
                details[i] = "x".join(map(str, detail))
            elif not isinstance(detail, str):
                raise TypeError(
                    f"Invalid layout value of '{detail}'. Must be a string or a tuple of ints."
                )
    elif param == "bondlength":
        for i, detail in enumerate(details):
            if isinstance(detail, float):
                details[i] = str(detail)
            elif isinstance(detail, int):
                details[i] = f"{detail:.1f}"
            elif not isinstance(detail, str):
                raise TypeError(f"Invalid bondlength '{detail}'. Must be a string, int or float.")

    for detail in details:
        if not isinstance(detail, str):
            raise TypeError(f"Invalid type '{type(detail).__name__}' for parameter '{param}'")

    return details

class Description(typing.Mapping[ParamName, ParamVal]):
    """An immutable and hashable dictionary that contains all the parameter
    values for a dataset."""

    def __init__(self, params: typing.Iterable[Tuple[ParamName, ParamVal]]):
        self.__data = dict(params)
        self.__hash = None

    def __getitem__(self, __key: ParamName) -> ParamVal:
        return self.__data[__key]

    def __iter__(self) -> typing.Iterator[ParamName]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __hash__(self) -> int:
        if not self.__hash:
            self.__hash = hash(tuple(self.__data))

        return self.__hash

    def __str__(self) -> str:
        return str(self.__data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.__data)})"

