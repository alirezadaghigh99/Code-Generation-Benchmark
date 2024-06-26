def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_dict = target_link.state_dict()
    source_dict = source_link.state_dict()
    for k, target_value in target_dict.items():
        source_value = source_dict[k]
        if source_value.dtype in [torch.float32, torch.float64, torch.float16]:
            assert target_value.shape == source_value.shape
            target_value.mul_(1 - tau)
            target_value.add_(tau * source_value)
        else:
            # Scalar type
            # Some modules such as BN has scalar value `num_batches_tracked`
            target_dict[k] = source_value