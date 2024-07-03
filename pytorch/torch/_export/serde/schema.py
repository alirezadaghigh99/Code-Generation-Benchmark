class Graph:
    inputs: List[Argument]
    outputs: List[Argument]
    nodes: List[Node]
    tensor_values: Dict[str, TensorMeta]
    sym_int_values: Dict[str, SymInt]
    sym_bool_values: Dict[str, SymBool]
    # This is for deserializing the submodule graphs from higher order ops
    # (ex. cond, map) where single tensor returns will just return a single
    # tensor, rather than following export schema and returning a singleton
    # list.
    is_single_tensor_return: bool = False
    custom_obj_values: Dict[str, CustomObjArgument] = field(default_factory=dict)