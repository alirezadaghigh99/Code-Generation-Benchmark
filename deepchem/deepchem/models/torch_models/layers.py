class MultilayerPerceptron(nn.Module):
    """A simple fully connected feed-forward network, otherwise known as a multilayer perceptron (MLP).

    Examples
    --------
    >>> model = MultilayerPerceptron(d_input=10, d_hidden=(2,3), d_output=2, dropout=0.0, activation_fn='relu')
    >>> x = torch.ones(2, 10)
    >>> out = model(x)
    >>> print(out.shape)
    torch.Size([2, 2])
    """

    def __init__(self,
                 d_input: int,
                 d_output: int,
                 d_hidden: Optional[tuple] = None,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 batch_norm_momentum: float = 0.1,
                 activation_fn: Union[Callable, str] = 'relu',
                 skip_connection: bool = False,
                 weighted_skip: bool = True):
        """Initialize the model.

        Parameters
        ----------
        d_input: int
            the dimension of the input layer
        d_output: int
            the dimension of the output layer
        d_hidden: tuple
            the dimensions of the hidden layers
        dropout: float
            the dropout probability
        batch_norm: bool
            whether to use batch normalization
        batch_norm_momentum: float
            the momentum for batch normalization
        activation_fn: str
            the activation function to use in the hidden layers
        skip_connection: bool
            whether to add a skip connection from the input to the output
        weighted_skip: bool
            whether to add a weighted skip connection from the input to the output
        """
        super(MultilayerPerceptron, self).__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.activation_fn = get_activation(activation_fn)
        self.model = nn.Sequential(*self.build_layers())
        self.skip = nn.Linear(d_input, d_output) if skip_connection else None
        self.weighted_skip = weighted_skip

    def build_layers(self):
        """
        Build the layers of the model, iterating through the hidden dimensions to produce a list of layers.
        """

        layer_list = []
        layer_dim = self.d_input
        if self.d_hidden is not None:
            for d in self.d_hidden:
                layer_list.append(nn.Linear(layer_dim, d))
                layer_list.append(self.dropout)
                if self.batch_norm:
                    layer_list.append(
                        nn.BatchNorm1d(d, momentum=self.batch_norm_momentum))
                layer_dim = d
        layer_list.append(nn.Linear(layer_dim, self.d_output))
        return layer_list

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        input = x
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = self.activation_fn(
                    x
                )  # Done because activation_fn returns a torch.nn.functional
        if self.skip is not None:
            if not self.weighted_skip:
                return x + input
            else:
                return x + self.skip(input)
        else:
            return xclass MolGANAggregationLayer(nn.Module):
    """
    Graph Aggregation layer used in MolGAN model.
    MolGAN is a WGAN type model for generation of small molecules.
    Performs aggregation on tensor resulting from convolution layers.
    Given its simple nature it might be removed in future and moved to
    MolGANEncoderLayer.


    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> vertices = 9
    >>> nodes = 5
    >>> edges = 5
    >>> units = 128

    >>> layer_1 = MolGANConvolutionLayer(units=units,nodes=nodes,edges=edges, name='layer1')
    >>> layer_2 = MolGANAggregationLayer(units=128, name='layer2')
    >>> adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    >>> node_tensor = torch.randn((1, vertices, nodes))
    >>> hidden_1 = layer_1([adjacency_tensor, node_tensor])
    >>> output = layer_2(hidden_1[2])

    References
    ----------
    .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
        for small molecular graphs", https://arxiv.org/abs/1805.11973
    """

    def __init__(self,
                 units: int = 128,
                 activation=torch.tanh,
                 dropout_rate: float = 0.0,
                 name: str = "",
                 prev_shape: int = 0,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize the layer

        Parameters
        ---------
        units: int, optional (default=128)
            Dimesion of dense layers used for aggregation
        activation: function, optional (default=Tanh)
            activation function used across model, default is Tanh
        dropout_rate: float, optional (default=0.0)
            Used by dropout layer
        name: string, optional (default="")
            Name of the layer
        prev_shape: int, optional (default=0)
            Shape of the input tensor
        """

        super(MolGANAggregationLayer, self).__init__()
        self.units: int = units
        self.activation = activation
        self.dropout_rate: float = dropout_rate
        self.name: str = name
        self.device = device

        if prev_shape:
            self.d1 = nn.Linear(prev_shape, self.units)
            self.d2 = nn.Linear(prev_shape, self.units)
        else:
            self.d1 = nn.Linear(self.units, self.units)
            self.d2 = nn.Linear(self.units, self.units)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def __repr__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        string
            String representation of the layer
        """
        return f"{self.__class__.__name__}(units={self.units}, activation={self.activation}, dropout_rate={self.dropout_rate}, Name={self.name})"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Invoke this layer

        Parameters
        ----------
        inputs: List
            Single tensor resulting from graph convolution layer

        Returns
        --------
        aggregation tensor: torch.Tensor
          Result of aggregation function on input convolution tensor.
        """
        inputs = inputs.to(self.device)
        i = torch.sigmoid(self.d1(inputs))
        j = self.activation(self.d2(inputs))
        output = torch.sum(i * j, dim=1)
        output = self.activation(output)
        output = self.dropout_layer(output)
        return outputclass MolGANMultiConvolutionLayer(nn.Module):
    """
    Multiple pass convolution layer used in MolGAN model.
    MolGAN is a WGAN type model for generation of small molecules.
    It takes outputs of previous convolution layer and uses
    them as inputs for the next one.
    It simplifies the overall framework, but might be moved to
    MolGANEncoderLayer in the future in order to reduce number of layers.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> vertices = 9
    >>> nodes = 5
    >>> edges = 5
    >>> units = (128,64)

    >>> layer_1 = MolGANMultiConvolutionLayer(units=units, nodes=nodes, edges=edges, name='layer1')
    >>> adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    >>> node_tensor = torch.randn((1, vertices, nodes))
    >>> output = layer_1([adjacency_tensor, node_tensor])

    References
    ----------
    .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
        for small molecular graphs", https://arxiv.org/abs/1805.11973
    """

    def __init__(self,
                 units: Tuple = (128, 64),
                 nodes: int = 5,
                 activation=torch.tanh,
                 dropout_rate: float = 0.0,
                 edges: int = 5,
                 name: str = "",
                 device: torch.device = torch.device('cpu'),
                 **kwargs):
        """
        Initialize the layer

        Parameters
        ---------
        units: Tuple, optional (default=(128,64)), min_length=2
            ist of dimensions used by consecutive convolution layers.
            The more values the more convolution layers invoked.
        nodes: int, optional (default=5)
            Number of features in node tensor
        activation: function, optional (default=Tanh)
            activation function used across model, default is Tanh
        dropout_rate: float, optional (default=0.0)
            Used by dropout layer
        edges: int, optional (default=5)
            Controls how many dense layers use for single convolution unit.
            Typically matches number of bond types used in the molecule.
        name: string, optional (default="")
            Name of the layer
        """

        super(MolGANMultiConvolutionLayer, self).__init__()
        if len(units) < 2:
            raise ValueError("units parameter must contain at least two values")

        self.nodes: int = nodes
        self.units: Tuple = units
        self.activation = activation
        self.dropout_rate: float = dropout_rate
        self.edges: int = edges
        self.name: str = name
        self.device = device

        self.first_convolution = MolGANConvolutionLayer(
            units=self.units[0],
            nodes=self.nodes,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            edges=self.edges,
            device=self.device)
        self.gcl = nn.ModuleList([
            MolGANConvolutionLayer(units=u,
                                   nodes=self.nodes,
                                   activation=self.activation,
                                   dropout_rate=self.dropout_rate,
                                   edges=self.edges,
                                   prev_shape=self.units[count],
                                   device=self.device)
            for count, u in enumerate(self.units[1:])
        ])

    def __repr__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        string
            String representation of the layer
        """
        return f"{self.__class__.__name__}(units={self.units}, nodes={self.nodes}, activation={self.activation}, dropout_rate={self.dropout_rate}), edges={self.edges}, Name={self.name})"

    def forward(self, inputs: List) -> torch.Tensor:
        """
        Invoke this layer

        Parameters
        ----------
        inputs: list
            List of two input matrices, adjacency tensor and node features tensors
            in one-hot encoding format.

        Returns
        --------
        convolution tensor: torch.Tensor
            Result of input tensors going through convolution a number of times.
        """

        adjacency_tensor = inputs[0].to(self.device)
        node_tensor = inputs[1].to(self.device)

        tensors = self.first_convolution([adjacency_tensor, node_tensor])

        # Loop over the remaining convolution layers
        for layer in self.gcl:
            # Apply the current layer to the outputs from the previous layer
            tensors = layer(tensors)

        _, _, hidden_tensor = tensors

        return hidden_tensorclass MolGANEncoderLayer(nn.Module):
    """
    Main learning layer used by MolGAN model.
    MolGAN is a WGAN type model for generation of small molecules.
    It role is to further simplify model.
    This layer can be manually built by stacking graph convolution layers
    followed by graph aggregation.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> vertices = 9
    >>> nodes = 5
    >>> edges = 5
    >>> dropout_rate = 0.0
    >>> adjacency_tensor = torch.randn((1, vertices, vertices, edges))
    >>> node_tensor = torch.randn((1, vertices, nodes))

    >>> graph = MolGANEncoderLayer(units = [(128,64),128], dropout_rate= dropout_rate, edges=edges, nodes=nodes)([adjacency_tensor,node_tensor])
    >>> dense = nn.Linear(128,128)(graph)
    >>> dense = torch.tanh(dense)
    >>> dense = nn.Dropout(dropout_rate)(dense)
    >>> dense = nn.Linear(128,64)(dense)
    >>> dense = torch.tanh(dense)
    >>> dense = nn.Dropout(dropout_rate)(dense)
    >>> output = nn.Linear(64,1)(dense)

    References
    ----------
    .. [1] Nicola De Cao et al. "MolGAN: An implicit generative model
        for small molecular graphs", https://arxiv.org/abs/1805.11973
    """

    def __init__(self,
                 units: List = [(128, 64), 128],
                 activation: Callable = torch.tanh,
                 dropout_rate: float = 0.0,
                 edges: int = 5,
                 nodes: int = 5,
                 name: str = "",
                 device: torch.device = torch.device('cpu'),
                 **kwargs):
        """
        Initialize the layer

        Parameters
        ----------
        units: List, optional (default=[(128,64),128])
            List of dimensions used by consecutive convolution layers.
            The more values the more convolution layers invoked.
        activation: function, optional (default=Tanh)
            activation function used across model, default is Tanh
        dropout_rate: float, optional (default=0.0)
            Used by dropout layer
        edges: int, optional (default=5)
            Controls how many dense layers use for single convolution unit.
            Typically matches number of bond types used in the molecule.
        nodes: int, optional (default=5)
            Number of features in node tensor
        name: string, optional (default="")
            Name of the layer
        """

        super(MolGANEncoderLayer, self).__init__()
        if len(units) != 2:
            raise ValueError("units parameter must contain two values")
        self.graph_convolution_units, self.auxiliary_units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.edges = edges
        self.nodes = nodes
        self.device = device

        self.multi_graph_convolution_layer = MolGANMultiConvolutionLayer(
            units=self.graph_convolution_units,
            nodes=self.nodes,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            edges=self.edges,
            device=self.device)
        self.graph_aggregation_layer = MolGANAggregationLayer(
            units=self.auxiliary_units,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            prev_shape=self.graph_convolution_units[-1] + nodes,
            device=self.device)

    def __repr__(self) -> str:
        """
        String representation of the layer

        Returns
        -------
        string
            String representation of the layer
        """
        return f"{self.__class__.__name__}(graph_convolution_units={self.graph_convolution_units}, auxiliary_units={self.auxiliary_units}, activation={self.activation}, dropout_rate={self.dropout_rate}), edges={self.edges})"

    def forward(self, inputs: List) -> torch.Tensor:
        """
        Invoke this layer

        Parameters
        ----------
        inputs: list
            List of two input matrices, adjacency tensor and node features tensors
            in one-hot encoding format.

        Returns
        --------
        encoder tensor: tf.Tensor
            Tensor that been through number of convolutions followed
            by aggregation.
        """

        output = self.multi_graph_convolution_layer(inputs)

        node_tensor = inputs[1]

        if len(inputs) > 2:
            hidden_tensor = inputs[2]
            annotations = torch.cat((output, hidden_tensor, node_tensor), -1)
        else:
            _, node_tensor = inputs
            annotations = torch.cat((output, node_tensor), -1)

        output = self.graph_aggregation_layer(annotations)
        return outputclass PositionwiseFeedForward(nn.Module):
    """PositionwiseFeedForward is a layer used to define the position-wise feed-forward (FFN) algorithm for the Molecular Attention Transformer [1]_

    Each layer in the MAT encoder contains a fully connected feed-forward network which applies two linear transformations and the given activation function.
    This is done in addition to the SublayerConnection module.

    Note: This modified version of `PositionwiseFeedForward` class contains `dropout_at_input_no_act` condition to facilitate its use in defining
        the feed-forward (FFN) algorithm for the Directed Message Passing Neural Network (D-MPNN) [2]_

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    .. [2] Analyzing Learned Molecular Representations for Property Prediction https://arxiv.org/pdf/1904.01561.pdf

    Examples
    --------
    >>> from deepchem.models.torch_models.layers import PositionwiseFeedForward
    >>> feed_fwd_layer = PositionwiseFeedForward(d_input = 2, d_hidden = 2, d_output = 2, activation = 'relu', n_layers = 1, dropout_p = 0.1)
    >>> input_tensor = torch.tensor([[1., 2.], [5., 6.]])
    >>> output_tensor = feed_fwd_layer(input_tensor)
  """

    def __init__(self,
                 d_input: int = 1024,
                 d_hidden: int = 1024,
                 d_output: int = 1024,
                 activation: str = 'leakyrelu',
                 n_layers: int = 1,
                 dropout_p: float = 0.0,
                 dropout_at_input_no_act: bool = False):
        """Initialize a PositionwiseFeedForward layer.

        Parameters
        ----------
        d_input: int
            Size of input layer.
        d_hidden: int (same as d_input if d_output = 0)
            Size of hidden layer.
        d_output: int (same as d_input if d_output = 0)
            Size of output layer.
        activation: str
            Activation function to be used. Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
        n_layers: int
            Number of layers.
        dropout_p: float
            Dropout probability.
        dropout_at_input_no_act: bool
            If true, dropout is applied on the input tensor. For single layer, it is not passed to an activation function.
        """
        super(PositionwiseFeedForward, self).__init__()

        self.dropout_at_input_no_act: bool = dropout_at_input_no_act

        if activation == 'relu':
            self.activation: Any = nn.ReLU()

        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif activation == 'prelu':
            self.activation = nn.PReLU()

        elif activation == 'tanh':
            self.activation = nn.Tanh()

        elif activation == 'selu':
            self.activation = nn.SELU()

        elif activation == 'elu':
            self.activation = nn.ELU()

        elif activation == "linear":
            self.activation = lambda x: x

        self.n_layers: int = n_layers
        d_output = d_output if d_output != 0 else d_input
        d_hidden = d_hidden if d_hidden != 0 else d_input

        if n_layers == 1:
            self.linears: Any = [nn.Linear(d_input, d_output)]

        else:
            self.linears = [nn.Linear(d_input, d_hidden)] + [
                nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 2)
            ] + [nn.Linear(d_hidden, d_output)]

        self.linears = nn.ModuleList(self.linears)
        dropout_layer = nn.Dropout(dropout_p)
        self.dropout_p = nn.ModuleList([dropout_layer for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output Computation for the PositionwiseFeedForward layer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        """
        if not self.n_layers:
            return x

        if self.n_layers == 1:
            if self.dropout_at_input_no_act:
                return self.linears[0](self.dropout_p[0](x))
            else:
                return self.dropout_p[0](self.activation(self.linears[0](x)))

        else:
            if self.dropout_at_input_no_act:
                x = self.dropout_p[0](x)
            for i in range(self.n_layers - 1):
                x = self.dropout_p[i](self.activation(self.linears[i](x)))
            return self.linears[-1](x)