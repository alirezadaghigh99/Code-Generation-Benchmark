class dMoE(torch.nn.Module):

    def __init__(
        self,
        device: Optional[torch.device],
        hidden_size: int = 1024,
        ffn_hidden_size: int = 4096,
        moe_num_experts: int = 1,
        moe_top_k: int = 1,
        mlp_type: str = 'mlp',
        activation_fn: Callable = DEFAULT_ACTIVATION_FN,
        moe_jitter_eps: Optional[float] = None,
        moe_normalize_expert_weights: Optional[Union[int, float]] = None,
        uniform_expert_assignment: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        # Token router.
        self.router = LearnedRouter(
            hidden_size,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_jitter_eps=moe_jitter_eps,
            moe_normalize_expert_weights=moe_normalize_expert_weights,
            uniform_expert_assignment=uniform_expert_assignment,
            device=device,
        )

        # Expert computation helper.
        self.experts = DroplessMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            mlp_type=mlp_type,
            moe_num_experts=moe_num_experts,
            activation_fn=activation_fn,
            bias=bias,
            device=device,
        )

    def forward(self, x: torch.Tensor):
        # Compute the expert scores and assignments.
        scores, expert_weights, top_experts = self.router(x)
        # Compute the experts.
        return self.experts(x, scores, expert_weights, top_experts)

