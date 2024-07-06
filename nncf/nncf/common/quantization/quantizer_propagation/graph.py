def is_insertion_point(qpsg_node_type: QuantizerPropagationStateGraphNodeType) -> bool:
        return qpsg_node_type in [
            QuantizerPropagationStateGraphNodeType.PRE_HOOK,
            QuantizerPropagationStateGraphNodeType.POST_HOOK,
        ]

