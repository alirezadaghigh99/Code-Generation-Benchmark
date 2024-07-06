def _chi_divergence(p: List[float], alpha: float = 1.0) -> float:
    assert alpha >= 1
    k = len(p)
    if not k:
        return -1.0
    return sum(abs(pi - 1.0 / k) ** alpha * k ** (alpha - 1.0) for pi in p)

def _kl_divergence(p: List[float]) -> float:
    k = len(p)
    if not k:
        return -1.0
    return sum(pi * math.log(k * pi) for pi in p if pi > 0)

