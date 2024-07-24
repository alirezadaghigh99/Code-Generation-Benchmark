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

class TensorArgument:
    name: str

class Node:
    target: str
    inputs: List[NamedArgument]
    outputs: List[Argument]
    metadata: Dict[str, str]

class InputSpec(_Union):
    user_input: UserInputSpec
    parameter: InputToParameterSpec
    buffer: InputToBufferSpec
    tensor_constant: InputToTensorConstantSpec
    custom_obj: InputToCustomObjSpec
    token: InputTokenSpec
    constant_input: ConstantInputSpec

class GraphModule:
    graph: Graph
    signature: GraphSignature
    # This is used for unflattening, by tracking the calling structure of all of
    # the modules in order to unflatten the modules back to the eager calling
    # conventions.
    module_call_graph: List[ModuleCallEntry]

