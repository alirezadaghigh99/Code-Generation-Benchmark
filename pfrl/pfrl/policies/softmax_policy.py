class SoftmaxCategoricalHead(nn.Module):
    def forward(self, logits):
        return torch.distributions.Categorical(logits=logits)

