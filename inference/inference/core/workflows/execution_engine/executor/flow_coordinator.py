def handle_execution_branch_selection(
    current_step: str,
    execution_graph: nx.DiGraph,
    selected_next_step: Optional[str],
) -> Set[str]:
    nodes_to_discard = set()
    if not execution_graph.has_node(selected_next_step):
        raise InvalidBlockBehaviourError(
            public_message=f"Block implementing step {current_step} requested flow control "
            f"mode `select_step`, but selected next step as: {selected_next_step} - which"
            f"is not a step that exists in workflow.",
            context="workflow_execution | flow_control_coordination",
        )
    for neighbour in execution_graph.neighbors(current_step):
        if execution_graph.nodes[neighbour].get("kind") != STEP_NODE_KIND:
            continue
        if neighbour == selected_next_step:
            continue
        neighbour_execution_path = get_all_nodes_in_execution_path(
            execution_graph=execution_graph, source=neighbour
        )
        nodes_to_discard.update(neighbour_execution_path)
    return nodes_to_discard