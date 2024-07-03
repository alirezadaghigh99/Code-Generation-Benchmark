def draw_graph(
    model: nn.Module,
    input_data: INPUT_DATA_TYPE | None = None,
    input_size: INPUT_SIZE_TYPE | None = None,
    graph_name: str = 'model',
    depth: int | float = 3,
    device: torch.device | str | None = None,
    dtypes: list[torch.dtype] | None = None,
    mode: str | None = None,
    strict: bool = True,
    expand_nested: bool = False,
    graph_dir: str | None = None,
    hide_module_functions: bool = True,
    hide_inner_tensors: bool = True,
    roll: bool = False,
    show_shapes: bool = True,
    save_graph: bool = False,
    filename: str | None = None,
    directory: str = '.',
    **kwargs: Any,
) -> ComputationGraph:
    '''Returns visual representation of the input Pytorch Module with
    ComputationGraph object. ComputationGraph object contains:

    1) Root nodes (usually tensor node for input tensors) which connect to all
    the other nodes of computation graph of pytorch module recorded during forward
    propagation.

    2) graphviz.Digraph object that contains visual representation of computation
    graph of pytorch module. This graph visual shows modules/ module hierarchy,
    torch_functions, shapes and tensors recorded during forward prop, for examples
    see documentation, and colab notebooks.


    Args:
        model (nn.Module):
            Pytorch model to represent visually.

        input_data (data structure containing torch.Tensor):
            input for forward method of model. Wrap it in a list for
            multiple args or in a dict or kwargs

        input_size (Sequence of Sizes):
            Shape of input data as a List/Tuple/torch.Size
            (dtypes must match model input, default is FloatTensors).
            Default: None

        graph_name (str):
            Name for graphviz.Digraph object. Also default name graphviz file
            of Graph Visualization
            Default: 'model'

        depth (int):
            Upper limit for depth of nodes to be shown in visualization.
            Depth is measured how far is module/tensor inside the module hierarchy.
            For instance, main module has depth=0, whereas submodule of main module
            has depth=1, and so on.
            Default: 3

        device (str or torch.device):
            Device to place and input tensors. Defaults to
            gpu if cuda is seen by pytorch, otherwise to cpu.
            Default: None

        dtypes (list of torch.dtype):
            Uses dtypes to set the types of input tensor if
            input size is given.

        mode (str):
            Mode of model to use for forward prop. Defaults
            to Eval mode if not given
            Default: None

        strict (bool):
            if true, graphviz visual does not allow multiple edges
            between nodes. Mutiple edge occurs e.g. when there are tensors
            from module node to module node and hiding those tensors
            Default: True

        expand_nested (bool):
            if true, shows nested modules with dashed borders

        graph_dir (str):
            Sets the direction of visual graph
            'TB' -> Top to Bottom
            'LR' -> Left to Right
            'BT' -> Bottom to Top
            'RL' -> Right to Left
            Default: None -> TB

        hide_module_function (bool):
            Determines whether to hide module torch_functions. Some
            modules consist only of torch_functions (no submodule),
            e.g. nn.Conv2d.
            True => Dont include module functions in graphviz
            False => Include modules function in graphviz
            Default: True

        hide_inner_tensors (bool):
            Inner tensor is all the tensors of computation graph
            but input and output tensors
            True => Does not show inner tensors in graphviz
            False => Shows inner tensors in graphviz
            Default: True

        roll (bool):
            If true, rolls recursive modules.
            Default: False

        show_shapes (bool):
            True => Show shape of tensor, input, and output
            False => Dont show
            Default: True

        save_graph (bool):
            True => Saves output file of graphviz graph
            False => Does not save
            Default: False

        filename (str):
            name of the file to store dot syntax representation and
            image file of graphviz graph. Defaults to graph_name

        directory (str):
            directory in which to store graphviz output files.
            Default: .

    Returns:
        ComputationGraph object that contains visualization of the input
        pytorch model in the form of graphviz Digraph object
    '''

    if filename is None:
        filename = f'{graph_name}.gv'

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode is None:
        model_mode = 'eval'
    else:
        model_mode = mode

    if graph_dir is None:
        graph_dir = 'TB'

    validate_user_params(
        model, input_data, input_size, depth, device, dtypes,
    )

    graph_attr = {
        'ordering': 'in',
        'rankdir': graph_dir,
    }

    # visual settings from torchviz
    # seems to work visually well
    node_attr = {
        'style': 'filled',
        'shape': 'plaintext',
        'align': 'left',
        'fontsize': '10',
        'ranksep': '0.1',
        'height': '0.2',
        'fontname': 'Linux libertine',
        'margin': '0',
    }

    edge_attr = {
        'fontsize': '10',
    }
    visual_graph = graphviz.Digraph(
        name=graph_name, engine='dot', strict=strict,
        graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr,
        directory=directory, filename=filename
    )

    input_recorder_tensor, kwargs_record_tensor, input_nodes = process_input(
        input_data, input_size, kwargs, device, dtypes
    )

    model_graph = ComputationGraph(
        visual_graph, input_nodes, show_shapes, expand_nested,
        hide_inner_tensors, hide_module_functions, roll, depth
    )

    forward_prop(
        model, input_recorder_tensor, device, model_graph,
        model_mode, **kwargs_record_tensor
    )

    model_graph.fill_visual_graph()

    if save_graph:
        model_graph.visual_graph.render(format='png')
    return model_graph