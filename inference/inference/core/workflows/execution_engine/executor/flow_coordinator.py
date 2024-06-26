def handle_flow_control(
    current_step_selector: str,
    flow_control: FlowControl,
    execution_graph: nx.DiGraph,
) -> Set[str]:
    nodes_to_discard = set()
    if flow_control.mode == "terminate_branch":
        nodes_to_discard = get_all_nodes_in_execution_path(
            execution_graph=execution_graph,
            source=current_step_selector,
            include_source=False,
        )
    elif flow_control.mode == "select_step":
        nodes_to_discard = handle_execution_branch_selection(
            current_step=current_step_selector,
            execution_graph=execution_graph,
            selected_next_step=flow_control.context,
        )
    return nodes_to_discard