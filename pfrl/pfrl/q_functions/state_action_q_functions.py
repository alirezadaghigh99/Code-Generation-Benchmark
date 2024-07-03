class FCLateActionSAQFunction(nn.Module, StateActionQFunction):
    """Fully-connected (s,a)-input Q-function with late action input.

    Actions are not included until the second hidden layer and not normalized.
    This architecture is used in the DDPG paper:
    http://arxiv.org/abs/1509.02971

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers. It must be greater than
            or equal to 1.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_channels,
        n_hidden_layers,
        nonlinearity=F.relu,
        last_wscale=1.0,
    ):
        assert n_hidden_layers >= 1
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity

        super().__init__()
        # No need to pass nonlinearity to obs_mlp because it has no
        # hidden layers
        self.obs_mlp = MLP(
            in_size=n_dim_obs, out_size=n_hidden_channels, hidden_sizes=[]
        )
        self.mlp = MLP(
            in_size=n_hidden_channels + n_dim_action,
            out_size=1,
            hidden_sizes=([self.n_hidden_channels] * (self.n_hidden_layers - 1)),
            nonlinearity=nonlinearity,
            last_wscale=last_wscale,
        )

        self.output = self.mlp.output

    def forward(self, state, action):
        h = self.nonlinearity(self.obs_mlp(state))
        h = torch.cat((h, action), dim=1)
        return self.mlp(h)