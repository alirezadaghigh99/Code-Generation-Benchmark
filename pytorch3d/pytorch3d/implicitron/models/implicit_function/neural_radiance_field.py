class NeuralRadianceFieldImplicitFunction(NeuralRadianceFieldBase):
    transformer_dim_down_factor: float = 1.0
    n_hidden_neurons_xyz: int = 256
    n_layers_xyz: int = 8
    append_xyz: Tuple[int, ...] = (5,)

    def _construct_xyz_encoder(self, input_dim: int):
        expand_args_fields(MLPWithInputSkips)
        return MLPWithInputSkips(
            self.n_layers_xyz,
            input_dim,
            self.n_hidden_neurons_xyz,
            input_dim,
            self.n_hidden_neurons_xyz,
            input_skips=self.append_xyz,
        )

