class JsonValidator:
    def __init__(self, expected_type: type):
        """
        JsonValidator(T) is function (s)->x that parses json string s into python value x, where x is of type T.

        Example Usage:
        >>> from typing import Union, List
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--x", type=JsonValidator(Union[int, List[float]]))
        >>> assert parser.parse_args(["--x=123"]).x == 123
        >>> assert parser.parse_args(["--x=[1, 2]"]).x == [1., 2.]
        """
        self.expected_type = expected_type

    @classmethod
    def _validate_and_cast(cls, json_value: Any, expected_type: Any):
        type_cls = (
            typing.get_origin(expected_type) or expected_type
        )  # typing.get_origin() returns None for non-generic types like `Any` and `int`
        type_args = typing.get_args(expected_type)
        if type_cls is typing.Any:
            return json_value
        if type_cls is float and isinstance(json_value, (int, float)):
            return float(json_value)
        elif type_cls in (int, str, bool) and isinstance(json_value, type_cls):
            return json_value
        elif type_cls is None and json_value is None:
            return None
        elif type_cls is typing.Union:
            for arg in type_args:
                try:
                    return cls._validate_and_cast(json_value, arg)
                except TypeError:
                    continue
        elif type_cls is dict and isinstance(json_value, dict):
            if not type_args:
                type_args = (Any, Any)
            type_key, type_value = type_args
            return {
                cls._validate_and_cast(key, type_key): cls._validate_and_cast(
                    value, type_value
                )
                for key, value in json_value.items()
            }
        elif type_cls is list and isinstance(json_value, list):
            if not type_args:
                type_args = [Any]
            return [cls._validate_and_cast(x, type_args[0]) for x in json_value]
        elif (
            type_cls is tuple
            and isinstance(json_value, list)
            and (type_args is None or len(type_args) == len(json_value))
        ):
            if type_args is None:
                type_args = [Any] * len(json_value)
            return tuple(
                type_cls(
                    cls._validate_and_cast(item, type_arg)
                    for item, type_arg in zip(json_value, type_args)
                )
            )
        raise TypeError(
            f"Cannot cast {json_value} with type {type(json_value)} to {expected_type}"
        )

    def __call__(self, str_value: str) -> Any:
        try:
            value = json.loads(str_value)
        except json.JSONDecodeError:
            raise TypeError(f"Cannot parse json value '{str_value}' for {self}")
        return self._validate_and_cast(value, self.expected_type)

    def __repr__(self):
        return f"JSON[{self.expected_type}]"

