class GraphData:
    """GraphData class

    This data class is almost same as `torch_geometric.data.Data
    <https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data>`_.

    Attributes
    ----------
    node_features: np.ndarray
        Node feature matrix with shape [num_nodes, num_node_features]
    edge_index: np.ndarray, dtype int
        Graph connectivity in COO format with shape [2, num_edges]
    edge_features: np.ndarray, optional (default None)
        Edge feature matrix with shape [num_edges, num_edge_features]
    node_pos_features: np.ndarray, optional (default None)
        Node position matrix with shape [num_nodes, num_dimensions].
    num_nodes: int
        The number of nodes in the graph
    num_node_features: int
        The number of features per node in the graph
    num_edges: int
        The number of edges in the graph
    num_edges_features: int, optional (default None)
        The number of features per edge in the graph

    Examples
    --------
    >>> import numpy as np
    >>> node_features = np.random.rand(5, 10)
    >>> edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    >>> edge_features = np.random.rand(5, 5)
    >>> global_features = np.random.random(5)
    >>> graph = GraphData(node_features, edge_index, edge_features, z=global_features)
    >>> graph
    GraphData(node_features=[5, 10], edge_index=[2, 5], edge_features=[5, 5], z=[5])
    """

    def __init__(self,
                 node_features: np.ndarray,
                 edge_index: np.ndarray,
                 edge_features: Optional[np.ndarray] = None,
                 node_pos_features: Optional[np.ndarray] = None,
                 **kwargs):
        """
        Parameters
        ----------
        node_features: np.ndarray
            Node feature matrix with shape [num_nodes, num_node_features]
        edge_index: np.ndarray, dtype int
            Graph connectivity in COO format with shape [2, num_edges]
        edge_features: np.ndarray, optional (default None)
            Edge feature matrix with shape [num_edges, num_edge_features]
        node_pos_features: np.ndarray, optional (default None)
            Node position matrix with shape [num_nodes, num_dimensions].
        kwargs: optional
            Additional attributes and their values
        """
        # validate params
        if isinstance(node_features, np.ndarray) is False:
            raise ValueError('node_features must be np.ndarray.')

        if isinstance(edge_index, np.ndarray) is False:
            raise ValueError('edge_index must be np.ndarray.')
        elif issubclass(edge_index.dtype.type, np.integer) is False:
            raise ValueError('edge_index.dtype must contains integers.')
        elif edge_index.shape[0] != 2:
            raise ValueError('The shape of edge_index is [2, num_edges].')

        # np.max() method works only for a non-empty array, so size of the array should be non-zero
        elif (edge_index.size != 0) and (np.max(edge_index) >=
                                         len(node_features)):
            raise ValueError('edge_index contains the invalid node number.')

        if edge_features is not None:
            if isinstance(edge_features, np.ndarray) is False:
                raise ValueError('edge_features must be np.ndarray or None.')
            elif edge_index.shape[1] != edge_features.shape[0]:
                raise ValueError(
                    'The first dimension of edge_features must be the same as the second dimension of edge_index.'
                )

        if node_pos_features is not None:
            if isinstance(node_pos_features, np.ndarray) is False:
                raise ValueError(
                    'node_pos_features must be np.ndarray or None.')
            elif node_pos_features.shape[0] != node_features.shape[0]:
                raise ValueError(
                    'The length of node_pos_features must be the same as the length of node_features.'
                )

        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.node_pos_features = node_pos_features
        self.kwargs = kwargs
        self.num_nodes, self.num_node_features = self.node_features.shape
        self.num_edges = edge_index.shape[1]
        if self.edge_features is not None:
            self.num_edge_features = self.edge_features.shape[1]

        for key, value in self.kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """Returns a string containing the printable representation of the object"""
        cls = self.__class__.__name__
        node_features_str = str(list(self.node_features.shape))
        edge_index_str = str(list(self.edge_index.shape))
        if self.edge_features is not None:
            edge_features_str = str(list(self.edge_features.shape))
        else:
            edge_features_str = "None"

        out = "%s(node_features=%s, edge_index=%s, edge_features=%s" % (
            cls, node_features_str, edge_index_str, edge_features_str)
        # Adding shapes of kwargs
        for key, value in self.kwargs.items():
            if isinstance(value, np.ndarray):
                out += (', ' + key + '=' + str(list(value.shape)))
            elif isinstance(value, str):
                out += (', ' + key + '=' + value)
            elif isinstance(value, int) or isinstance(value, float):
                out += (', ' + key + '=' + str(value))
        out += ')'
        return out

    def to_pyg_graph(self):
        """Convert to PyTorch Geometric graph data instance

        Returns
        -------
        torch_geometric.data.Data
            Graph data for PyTorch Geometric

        Note
        ----
        This method requires PyTorch Geometric to be installed.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ModuleNotFoundError:
            raise ImportError(
                "This function requires PyTorch Geometric to be installed.")

        edge_features = self.edge_features
        if edge_features is not None:
            edge_features = torch.from_numpy(self.edge_features).float()
        node_pos_features = self.node_pos_features
        if node_pos_features is not None:
            node_pos_features = torch.from_numpy(self.node_pos_features).float()
        kwargs = {}
        for key, value in self.kwargs.items():
            kwargs[key] = torch.from_numpy(value).float()
        return Data(x=torch.from_numpy(self.node_features).float(),
                    edge_index=torch.from_numpy(self.edge_index).long(),
                    edge_attr=edge_features,
                    pos=node_pos_features,
                    **kwargs)

    def to_dgl_graph(self, self_loop: bool = False):
        """Convert to DGL graph data instance

        Returns
        -------
        dgl.DGLGraph
            Graph data for DGL
        self_loop: bool
            Whether to add self loops for the nodes, i.e. edges from nodes
            to themselves. Default to False.

        Note
        ----
        This method requires DGL to be installed.
        """
        try:
            import dgl
            import torch
        except ModuleNotFoundError:
            raise ImportError("This function requires DGL to be installed.")

        src = self.edge_index[0]
        dst = self.edge_index[1]

        g = dgl.graph(
            (torch.from_numpy(src).long(), torch.from_numpy(dst).long()),
            num_nodes=self.num_nodes)
        g.ndata['x'] = torch.from_numpy(self.node_features).float()

        if self.node_pos_features is not None:
            g.ndata['pos'] = torch.from_numpy(self.node_pos_features).float()
            g.edata['d'] = torch.norm(g.ndata['pos'][g.edges()[0]] -
                                      g.ndata['pos'][g.edges()[1]],
                                      p=2,
                                      dim=-1).unsqueeze(-1).detach()
        if self.edge_features is not None:
            g.edata['edge_attr'] = torch.from_numpy(self.edge_features).float()

        if self_loop:
            # This assumes that the edge features for self loops are full-zero tensors
            # In the future we may want to support featurization for self loops
            g.add_edges(np.arange(self.num_nodes), np.arange(self.num_nodes))

        return g

    def numpy_to_torch(self, device: str = 'cpu'):
        """Convert numpy arrays to torch tensors. This may be useful when you are using PyTorch Geometric with GraphData objects.

        Parameters
        ----------
        device : str
            Device to store the tensors. Default to 'cpu'.

        Example
        -------
        >>> num_nodes, num_node_features = 5, 32
        >>> num_edges, num_edge_features = 6, 32
        >>> node_features = np.random.random_sample((num_nodes, num_node_features))
        >>> edge_features = np.random.random_sample((num_edges, num_edge_features))
        >>> edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        >>> graph_data = GraphData(node_features, edge_index, edge_features)
        >>> graph_data = graph_data.numpy_to_torch()
        >>> print(type(graph_data.node_features))
        <class 'torch.Tensor'>
        """
        import copy

        import torch
        graph_copy = copy.deepcopy(self)

        graph_copy.node_features = torch.from_numpy(  # type: ignore
            self.node_features).float().to(device)
        graph_copy.edge_index = torch.from_numpy(  # type: ignore
            self.edge_index).long().to(device)
        if self.edge_features is not None:
            graph_copy.edge_features = torch.from_numpy(  # type: ignore
                self.edge_features).float().to(device)
        else:
            graph_copy.edge_features = None
        if self.node_pos_features is not None:
            graph_copy.node_pos_features = torch.from_numpy(  # type: ignore
                self.node_pos_features).float().to(device)
        else:
            graph_copy.node_pos_features = None

        graph_copy.kwargs = {}
        for key, value in self.kwargs.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).to(device)
                graph_copy.kwargs[key] = value
                setattr(graph_copy, key, value)

        return graph_copy

    def subgraph(self, nodes):
        """Returns a subgraph of `nodes` indicies.

        Parameters
        ----------
        nodes : list, iterable
            A list of node indices to be included in the subgraph.

        Returns
        -------
        subgraph_data : GraphData
            A new GraphData object containing the subgraph induced on `nodes`.

        Example
        -------
        >>> import numpy as np
        >>> from deepchem.feat.graph_data import GraphData
        >>> node_features = np.random.rand(5, 10)
        >>> edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
        >>> edge_features = np.random.rand(5, 3)
        >>> graph_data = GraphData(node_features, edge_index, edge_features)
        >>> nodes = [0, 2, 4]
        >>> subgraph_data, node_mapping = graph_data.subgraph(nodes)
        """
        nodes = set(nodes)
        if not nodes.issubset(range(self.num_nodes)):
            raise ValueError("Some nodes are not in the original graph")

        # Create a mapping from the original node indices to the new node indices
        node_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(nodes)
        }

        # Filter and reindex node features
        subgraph_node_features = self.node_features[list(nodes)]

        # Filter and reindex edge indices and edge features
        subgraph_edge_indices = []
        subgraph_edge_features = []
        if self.edge_features is not None:
            for i in range(self.num_edges):
                src, dest = self.edge_index[:, i]
                if src in nodes and dest in nodes:
                    subgraph_edge_indices.append(
                        (node_mapping[src], node_mapping[dest]))
                    subgraph_edge_features.append(self.edge_features[i])

        subgraph_edge_index = np.array(subgraph_edge_indices, dtype=np.int64).T
        subgraph_edge_features = np.array(subgraph_edge_features)

        subgraph_data = GraphData(node_features=subgraph_node_features,
                                  edge_index=subgraph_edge_index,
                                  edge_features=subgraph_edge_features,
                                  **self.kwargs)

        return subgraph_data, node_mapping