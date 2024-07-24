class FPSSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
    num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def sample(self, pos, batch, **kwargs):
        from torch_geometric.nn import fps

        if len(pos.shape) != 2:
            raise ValueError(" This class is for sparse data and expects the pos tensor to be of dimension 2")
        return fps(pos, batch, ratio=self._get_ratio_to_sample(pos.shape[0]))

class RandomSampler(BaseSampler):
    """If num_to_sample is provided, sample exactly
    num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def sample(self, pos, batch, **kwargs):
        if len(pos.shape) != 2:
            raise ValueError(" This class is for sparse data and expects the pos tensor to be of dimension 2")
        idx = torch.randperm(pos.shape[0])[
            0 : self._get_num_to_sample(pos.shape[0]),
        ]
        return idx

