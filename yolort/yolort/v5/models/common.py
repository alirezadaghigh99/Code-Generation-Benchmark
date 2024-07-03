def space_to_depth(x: Tensor) -> Tensor:
    """x(b,c,w,h) -> y(b,4c,w/2,h/2)"""
    N, C, H, W = x.size()
    x = x.reshape(N, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 5, 3, 1, 2, 4)
    y = x.reshape(N, C * 4, H // 2, W // 2)
    return y