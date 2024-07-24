class ParamItem(NamedTuple):
    name: str
    data: Optional[Union[Dict[str, Tensor], List["ParamItem"]]]

