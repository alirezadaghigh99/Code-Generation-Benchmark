class ContEmbeddings(nn.Module):
    def __init__(
        self,
        n_cont_cols: int,
        embed_dim: int,
        embed_dropout: float,
        full_embed_dropout: bool,
        activation_fn: Optional[str] = None,
    ):
        super(ContEmbeddings, self).__init__()

        self.n_cont_cols = n_cont_cols
        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout
        self.activation_fn_name = activation_fn

        self.weight = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim))

        self.reset_parameters()

        self.activation_fn = (
            get_activation_fn(activation_fn) if activation_fn is not None else None
        )

        if full_embed_dropout:
            self.dropout: DropoutLayers = FullEmbeddingDropout(embed_dropout)
        else:
            self.dropout = nn.Dropout(embed_dropout)

    def reset_parameters(self) -> None:
        # see here https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: Tensor) -> Tensor:
        # same as torch.einsum("ij,jk->ijk", X, weight)
        x = self.weight.unsqueeze(0) * X.unsqueeze(2)
        x = x + self.bias.unsqueeze(0)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.dropout(x)
        return x

    def extra_repr(self) -> str:
        all_params = "INFO: [ContLinear = weight(n_cont_cols, embed_dim) + bias(n_cont_cols, embed_dim)]\n"
        all_params += (
            "(linear): ContLinear(n_cont_cols={n_cont_cols}, embed_dim={embed_dim}"
        )
        all_params += ", embed_dropout={embed_dropout})"
        return f"{all_params.format(**self.__dict__)}"

class FullEmbeddingDropout(nn.Module):
    def __init__(self, p: float):
        super(FullEmbeddingDropout, self).__init__()

        if p < 0 or p > 1:
            raise ValueError(f"p probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            mask = X.new().resize_((X.size(1), 1)).bernoulli_(1 - self.p).expand_as(
                X
            ) / (1 - self.p)
            return mask * X
        else:
            return X

    def extra_repr(self) -> str:
        return f"p={self.p}"

class SharedEmbeddings(nn.Module):
    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        add_shared_embed: bool = False,
        frac_shared_embed=0.25,
    ):
        super(SharedEmbeddings, self).__init__()

        assert frac_shared_embed < 1, "'frac_shared_embed' must be less than 1"
        self.add_shared_embed = add_shared_embed
        self.embed = nn.Embedding(n_embed, embed_dim, padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)
        if add_shared_embed:
            col_embed_dim = embed_dim
        else:
            col_embed_dim = int(embed_dim * frac_shared_embed)
        self.shared_embed = nn.Parameter(torch.empty(1, col_embed_dim).uniform_(-1, 1))

    def forward(self, X: Tensor) -> Tensor:
        out = self.embed(X)
        shared_embed = self.shared_embed.expand(out.shape[0], -1)
        if self.add_shared_embed:
            out += shared_embed
        else:
            out[:, : shared_embed.shape[1]] = shared_embed
        return out

