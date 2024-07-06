def compare_nx_graph_with_reference(
    nx_graph: nx.DiGraph,
    path_to_dot: str,
    sort_dot_graph: bool = True,
    check_edge_attrs: bool = False,
    unstable_node_names: bool = False,
) -> None:
    """
    Checks whether the two nx.DiGraph are identical. The first one is 'nx_graph' argument
    and the second graph is read from the absolute path - 'path_to_dot'.
    Also, could dump the graph, based in the global variable NNCF_TEST_REGEN_DOT.
    If 'sort_dot_graph' is True sorts the second graph before dumping.
    If 'check_edge_attrs' is True checks edge attributes of the graphs.
    :param nx_graph: The first nx.DiGraph.
    :param path_to_dot: The absolute path to the second nx.DiGraph.
    :param sort_dot_graph: whether to call sort_dot() function on the second graph.
    :param check_edge_attrs: whether to check edge attributes of the graphs.
    :return: None
    """
    dot_dir = Path(path_to_dot).parent
    # validate .dot file manually!
    if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
        if not os.path.exists(dot_dir):
            os.makedirs(dot_dir)
        write_dot_graph(nx_graph, Path(path_to_dot))
        if sort_dot_graph:
            sort_dot(path_to_dot)

    expected_graph = nx.DiGraph(read_dot_graph(Path(path_to_dot)))
    check_nx_graph(nx_graph, expected_graph, check_edge_attrs, unstable_node_names)

