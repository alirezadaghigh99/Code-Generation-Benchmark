class TransformerDecoderLayer(nn.Module):
            def __init__(self, d_model, nhead, batch_first):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=batch_first)

