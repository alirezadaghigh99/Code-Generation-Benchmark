def needs_packages(*names: str) -> unittest.skipIf:
    """Decorate a test such that it only runs if the rqeuired package is available."""
    missing = {name for name in names if not is_installed(name=name)}
    return unittest.skipIf(condition=missing, reason=f"Missing required packages: {sorted(missing)}.")

def rand(*size: int, generator: torch.Generator, device: torch.device) -> torch.FloatTensor:
    """Wrap generating random numbers with a generator and given device."""
    return torch.rand(*size, generator=generator).to(device=device)

