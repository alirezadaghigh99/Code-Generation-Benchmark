class NNCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = NNCLRProjectionHead(512, 512, 128)
        self.prediction_head = NNCLRPredictionHead(128, 512, 128)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

