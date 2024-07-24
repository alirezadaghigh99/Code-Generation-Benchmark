class WordDropout(nn.Module):
    """ A word dropout layer that's designed for embedded inputs (e.g., any inputs to an LSTM layer).
    Given a batch of embedded inputs, this layer randomly set some of them to be a replacement state.
    Note that this layer assumes the last dimension of the input to be the hidden dimension of a unit.
    """
    def __init__(self, dropprob):
        super().__init__()
        self.dropprob = dropprob

    def forward(self, x, replacement=None):
        if not self.training or self.dropprob == 0:
            return x

        masksize = [y for y in x.size()]
        masksize[-1] = 1
        dropmask = torch.rand(*masksize, device=x.device) < self.dropprob

        res = x.masked_fill(dropmask, 0)
        if replacement is not None:
            res = res + dropmask.float() * replacement

        return res
    
    def extra_repr(self):
        return 'p={}'.format(self.dropprob)

