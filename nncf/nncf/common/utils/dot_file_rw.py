def write_dot_graph(G: nx.DiGraph, path: pathlib.Path) -> None:
    # NOTE: writing dot files with colons even in labels or other node/edge/graph attributes leads to an
    # error. See https://github.com/networkx/networkx/issues/5962. If `relabel` is True in this function,
    # then the colons (:) will be replaced with (^) symbols.
    relabeled = relabel_graph_for_dot_visualization(G)
    nx.nx_pydot.write_dot(relabeled, str(path))

