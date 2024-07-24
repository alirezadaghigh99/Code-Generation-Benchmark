class NNCFGraphToTestConstantFiltering:
    def __init__(
        self,
        constant_metatype,
        node_with_weights_metatype,
        concat_layer_attr,
        add_node_between_const_and_weight_node,
        nncf_graph_cls=NNCFGraph,
    ) -> None:
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv", node_with_weights_metatype),
            NodeWithType("Weights", constant_metatype),
            NodeWithType("Weights2", constant_metatype),
            NodeWithType("Conv2", node_with_weights_metatype),
            NodeWithType("ReadVariable", InputNoopMetatype),
            NodeWithType("Add", None),
            NodeWithType("Weights3", constant_metatype),
            NodeWithType("Weights4", constant_metatype),
            NodeWithType("Conv3", node_with_weights_metatype),
            NodeWithType("NodeAfterConstantConv", None),
            NodeWithType("Final_node", None),
            NodeWithType("Input_2", InputNoopMetatype),
            NodeWithType("Const0", constant_metatype),
            NodeWithType("Const1", constant_metatype),
            NodeWithType("Concat_with_input", None, layer_attributes=concat_layer_attr),
            NodeWithType("Const2", constant_metatype),
            NodeWithType("Const3", constant_metatype),
            NodeWithType("Const4", constant_metatype),
            NodeWithType("Concat_with_constant", None, layer_attributes=concat_layer_attr),
            NodeWithType("Const5", constant_metatype),
            NodeWithType("Const6", constant_metatype),
            NodeWithType("Concat_with_missed_input", None, layer_attributes=concat_layer_attr),
        ]

        edges = [
            ("Input_1", "Conv"),
            ("Weights", "Conv"),
            ("Weights2", "Conv2"),
            ("Conv2", "Add"),
            ("ReadVariable", "Add"),
            ("Add", "Final_node"),
            ("Weights3", "Conv3"),
            ("Weights4", "Conv3"),
            ("Conv3", "NodeAfterConstantConv"),
            ("Input_2", "Concat_with_input"),
            ("Const0", "Concat_with_input"),
            ("Const1", "Concat_with_input"),
            ("Const2", "Concat_with_constant"),
            ("Const3", "Concat_with_constant"),
            ("Const4", "Concat_with_constant"),
            ("Const5", "Concat_with_missed_input"),
            ("Const6", "Concat_with_missed_input"),
        ]
        if add_node_between_const_and_weight_node:
            constant_nodes = [node for node in nodes if node.node_op_metatype is constant_metatype]
            const_node_to_edge = {}
            for node in constant_nodes:
                for i, edge in enumerate(edges):
                    if node.node_name == edge[0]:
                        const_node_to_edge[node] = edge
                        break
                del edges[i]
            for node, edge in const_node_to_edge.items():
                any_after_node_name = f"AnyAfter{node.node_name}"
                nodes.append(NodeWithType(any_after_node_name, None))
                edges.append((edge[0], any_after_node_name))
                edges.append((any_after_node_name, edge[1]))

        original_mock_graph = create_mock_graph(nodes, edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)

class NNCFGraphCA:
    def __init__(
        self,
        conv_metatype,
        conv_layer_attrs=None,
        conv_2_layer_attrs=None,
        use_one_layer_attrs=True,
        nncf_graph_cls=NNCFGraph,
    ):
        #       Original graph
        #          Input_1
        #             |
        #           Conv_1
        #             |
        #           Conv_2
        #             |
        #           Output_1
        if use_one_layer_attrs and conv_layer_attrs is not None and conv_2_layer_attrs is None:
            conv_2_layer_attrs = conv_layer_attrs
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1_W", ConstantTestMetatype),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Conv_2_W", ConstantTestMetatype),
            NodeWithType("Conv_2", conv_metatype, layer_attributes=conv_2_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Conv_2"),
            ("Conv_2", "Output_1"),
            ("Conv_1_W", "Conv_1"),
            ("Conv_2_W", "Conv_2"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)

class NNCFGraphCAWithBias:
    def __init__(
        self,
        conv_metatype,
        add_metatype,
        conv_1_layer_attrs=None,
        conv_2_layer_attrs=None,
        both_biases=True,
        add_layer_attrs=None,
        constant_metatype=ConstantTestMetatype,
        nncf_graph_cls=NNCFGraph,
    ):
        #       Original graph
        #          Input_1
        #             |
        #           Conv_1
        #             |
        #           Add_1
        #             |
        #           Conv_2
        #             |
        #           Add_2
        #           Output_1
        if conv_2_layer_attrs is None:
            conv_2_layer_attrs = conv_1_layer_attrs
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1_W", constant_metatype),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_1_layer_attrs),
            NodeWithType("Add_1_W", constant_metatype),
            NodeWithType("Add_1", add_metatype, layer_attributes=add_layer_attrs),
            NodeWithType("Conv_2_W", constant_metatype),
            NodeWithType("Conv_2", conv_metatype, layer_attributes=conv_2_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        if both_biases:
            nodes.extend(
                [
                    NodeWithType("Add_2_W", constant_metatype),
                    NodeWithType("Add_2", add_metatype, layer_attributes=add_layer_attrs),
                ]
            )
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Add_1"),
            ("Add_1", "Conv_2"),
            ("Conv_1_W", "Conv_1"),
            ("Add_1_W", "Add_1"),
            ("Conv_2_W", "Conv_2"),
        ]
        if both_biases:
            node_edges.extend([("Conv_2", "Add_2"), ("Add_2", "Output_1"), ("Add_2_W", "Add_2")])
        else:
            node_edges.extend([("Conv_2", "Output_1")])
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)

