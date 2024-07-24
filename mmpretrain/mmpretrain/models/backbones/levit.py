class Attention(BaseModule):

    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            act_cfg=dict(type='HSwish'),
            resolution=14,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = LinearBatchNorm(dim, h)
        self.proj = nn.Sequential(
            build_activation_layer(act_cfg), LinearBatchNorm(self.dh, dim))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """change the mode of model."""
        super(Attention, self).train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape  # 2 196 128
        qkv = self.qkv(x)  # 2 196 128
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d],
            dim=3)  # q 2 196 4 16 ; k 2 196 4 16; v 2 196 4 32
        q = q.permute(0, 2, 1, 3)  # 2 4 196 16
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = ((q @ k.transpose(-2, -1)) *
                self.scale  # 2 4 196 16 * 2 4 16 196 -> 2 4 196 196
                + (self.attention_biases[:, self.attention_bias_idxs]
                   if self.training else self.ab))
        attn = attn.softmax(dim=-1)  # 2 4 196 196 -> 2 4 196 196
        x = (attn @ v).transpose(1, 2).reshape(
            B, N,
            self.dh)  # 2 4 196 196 * 2 4 196 32 -> 2 4 196 32 -> 2 196 128
        x = self.proj(x)
        return x

