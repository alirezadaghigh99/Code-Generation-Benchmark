class DistributionalFCStateQFunctionWithDiscreteAction(
    DistributionalSingleModelStateQFunctionWithDiscreteAction
):
    """Distributional fully-connected Q-function with discrete actions.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_actions (int): Number of actions in action space.
        n_atoms (int): Number of atoms of return distribution.
        v_min (float): Minimum value this model can approximate.
        v_max (float): Maximum value this model can approximate.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity applied after each hidden layer.
        last_wscale (float): Weight scale of the last layer.
    """

    def __init__(
        self,
        ndim_obs,
        n_actions,
        n_atoms,
        v_min,
        v_max,
        n_hidden_channels,
        n_hidden_layers,
        nonlinearity=F.relu,
        last_wscale=1.0,
    ):
        assert n_atoms >= 2
        assert v_min < v_max
        z_values = np.linspace(v_min, v_max, num=n_atoms, dtype=np.float32)
        model = nn.Sequential(
            MLP(
                in_size=ndim_obs,
                out_size=n_actions * n_atoms,
                hidden_sizes=[n_hidden_channels] * n_hidden_layers,
                nonlinearity=nonlinearity,
                last_wscale=last_wscale,
            ),
            Lambda(lambda x: torch.reshape(x, (-1, n_actions, n_atoms))),
            nn.Softmax(dim=2),
        )
        super().__init__(model=model, z_values=z_values)