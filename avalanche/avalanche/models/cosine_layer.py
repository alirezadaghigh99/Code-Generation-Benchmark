class CosineLinear(nn.Module):
    """
    Cosine layer defined in
    "Learning a Unified Classifier Incrementally via Rebalancing"
    by Saihui Hou et al.

    Implementation modified from https://github.com/G-U-N/PyCIL

    This layer is aimed at countering the task-recency bias by removing the bias
    in the classifier and normalizing the weight and the input feature before
    computing the weight-feature product
    """

    def __init__(self, in_features, out_features, sigma=True):
        """
        :param in_features: number of input features
        :param out_features: number of classes
        :param sigma: learnable output scaling factor
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter("sigma", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(
            F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1)
        )
        if self.sigma is not None:
            out = self.sigma * out

        return out

class SplitCosineLinear(nn.Module):
    """
    This class keeps two Cosine Linear layers, without sigma scaling,
    and handles the sigma parameter that is common for the two of them.
    One CosineLinear is for the old classes and the other
    one is for the new classes
    """

    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter("sigma", None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1, out2), dim=1)

        if self.sigma is not None:
            out = self.sigma * out

        return out

