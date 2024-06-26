def _flatten_and_concat_variables(vs):
    """Flatten and concat variables to make a single flat vector variable."""
    return torch.cat([torch.flatten(v) for v in vs], dim=0)