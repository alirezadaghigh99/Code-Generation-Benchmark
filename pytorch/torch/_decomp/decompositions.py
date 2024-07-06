def baddbmm(self, batch1, batch2, beta=1, alpha=1):
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    result = torch.bmm(batch1, batch2)
    if not isinstance(alpha, numbers.Number) or alpha != 1:
        result = result * alpha
    if beta == 0:
        return result
    if not isinstance(beta, numbers.Number) or beta != 1:
        self = self * beta
    return self + result

def mv(self, vec):
    torch._check(
        self.dim() == 2 and vec.dim() == 1,
        lambda: f"matrix @ vector expected, got {self.dim()}, {vec.dim()}",
    )
    torch._check(
        self.size(1) == vec.size(0),
        lambda: f"size mismatch, got input ({self.size(0)}x{self.size(1)}), vec ({vec.size(0)})",
    )
    return (self * vec).sum(dim=1)

