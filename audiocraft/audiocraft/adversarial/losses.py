def get_adv_criterion(loss_type: str) -> tp.Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == 'mse':
        return mse_loss
    elif loss_type == 'hinge':
        return hinge_loss
    elif loss_type == 'hinge2':
        return hinge2_loss
    raise ValueError('Unsupported loss')

class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for adversarial training.

    Args:
        loss (nn.Module): Loss to use for feature matching (default=torch.nn.L1).
        normalize (bool): Whether to normalize the loss.
            by number of feature maps.
    """
    def __init__(self, loss: nn.Module = torch.nn.L1Loss(), normalize: bool = True):
        super().__init__()
        self.loss = loss
        self.normalize = normalize

    def forward(self, fmap_fake: tp.List[torch.Tensor], fmap_real: tp.List[torch.Tensor]) -> torch.Tensor:
        assert len(fmap_fake) == len(fmap_real) and len(fmap_fake) > 0
        feat_loss = torch.tensor(0., device=fmap_fake[0].device)
        feat_scale = torch.tensor(0., device=fmap_fake[0].device)
        n_fmaps = 0
        for (feat_fake, feat_real) in zip(fmap_fake, fmap_real):
            assert feat_fake.shape == feat_real.shape
            n_fmaps += 1
            feat_loss += self.loss(feat_fake, feat_real)
            feat_scale += torch.mean(torch.abs(feat_real))

        if self.normalize:
            feat_loss /= n_fmaps

        return feat_loss

