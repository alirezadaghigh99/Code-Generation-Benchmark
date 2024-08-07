class MLP(nn.Module):
    r"""A Multi-Layer Perceptron with hidden layers.

    This model uses the MLP block with output dimension 1, which is squeezed.

    This combines multiple dense layers and activations functions into a single object.
    It separates the output layer from the hidden layers,
    since it typically has a different form.
    One can specify the specific activation functions per layer.
    The size of the hidden dimensions can be provided as a number,
    or as a factor relative to the input size (similar as for RBM).
    The default model is a single linear layer without activations.

    Forms a common building block for models such as
    `PauliNet (continuous) <https://www.nature.com/articles/s41557-020-0544-y>`_
    """
    hidden_dims: Optional[Union[int, tuple[int, ...]]] = None
    """The size of the hidden layers, excluding the output layer."""
    hidden_dims_alpha: Optional[Union[int, tuple[int, ...]]] = None
    """The size of the hidden layers provided as number of times the input size.
    One must choose to either specify this or the hidden_dims keyword argument"""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    hidden_activations: Optional[Union[Callable, tuple[Callable, ...]]] = nknn.gelu
    """The nonlinear activation function after each hidden layer.
    Can be provided as a single activation,
    where the same activation will be used for every layer."""
    output_activation: Optional[Callable] = None
    """The nonlinear activation at the output layer.
    If None is provided, the output layer will be essentially linear."""
    use_hidden_bias: bool = True
    """if True uses a bias in the hidden layer."""
    use_output_bias: bool = False
    """if True adds a bias to the output layer."""
    precision: Optional[jax.lax.Precision] = None
    """Numerical precision of the computation see :class:`jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = default_bias_init
    """Initializer for the biases."""

    @nn.compact
    def __call__(self, input):
        x = nknn.blocks.MLP(
            output_dim=1,  # a netket model has a single output
            hidden_dims=self.hidden_dims,
            hidden_dims_alpha=self.hidden_dims_alpha,
            param_dtype=self.param_dtype,
            hidden_activations=self.hidden_activations,
            output_activation=self.output_activation,
            use_hidden_bias=self.use_hidden_bias,
            use_output_bias=self.use_output_bias,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(input)
        x = x.squeeze(-1)
        return x

