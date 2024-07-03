class DenseSIFTDescriptor(Module):
    """Module, which computes SIFT descriptor densely over the image.

    Args:
        num_ang_bins: Number of angular bins. (8 is default)
        num_spatial_bins: Number of spatial bins per descriptor (4 is default).
    You might want to set odd number and relevant padding to keep feature map size
        spatial_bin_size: Size of a spatial bin in pixels (4 is default)
        clipval: clipping value to reduce single-bin dominance
        rootsift: (bool) if True, RootSIFT (Arandjelović et. al, 2012) is computed
        stride: default 1
        padding: default 0

    Returns:
        Tensor: DenseSIFT descriptor of the image

    Shape:
        - Input: (B, 1, H, W)
        - Output: (B, num_ang_bins * num_spatial_bins ** 2, (H+padding)/stride, (W+padding)/stride)

    Examples::
        >>> input =  torch.rand(2, 1, 200, 300)
        >>> SIFT = DenseSIFTDescriptor()
        >>> descs = SIFT(input) # 2x128x194x294
    """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_ang_bins={self.num_ang_bins}, "
            f"num_spatial_bins={self.num_spatial_bins}, "
            f"spatial_bin_size={self.spatial_bin_size}, "
            f"rootsift={self.rootsift}, "
            f"stride={self.stride}, "
            f"clipval={self.clipval})"
        )

    def __init__(
        self,
        num_ang_bins: int = 8,
        num_spatial_bins: int = 4,
        spatial_bin_size: int = 4,
        rootsift: bool = True,
        clipval: float = 0.2,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.eps = 1e-10
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.spatial_bin_size = spatial_bin_size
        self.clipval = clipval
        self.rootsift = rootsift
        self.stride = stride
        self.pad = padding
        nw = get_sift_pooling_kernel(ksize=self.spatial_bin_size).float()
        self.bin_pooling_kernel = nn.Conv2d(
            1,
            1,
            kernel_size=(nw.size(0), nw.size(1)),
            stride=(1, 1),
            bias=False,
            padding=(nw.size(0) // 2, nw.size(1) // 2),
        )
        self.bin_pooling_kernel.weight.data.copy_(nw.reshape(1, 1, nw.size(0), nw.size(1)))
        self.PoolingConv = nn.Conv2d(
            num_ang_bins,
            num_ang_bins * num_spatial_bins**2,
            kernel_size=(num_spatial_bins, num_spatial_bins),
            stride=(self.stride, self.stride),
            bias=False,
            padding=(self.pad, self.pad),
        )
        self.PoolingConv.weight.data.copy_(
            _get_reshape_kernel(num_ang_bins, num_spatial_bins, num_spatial_bins).float()
        )

    def get_pooling_kernel(self) -> Tensor:
        return self.bin_pooling_kernel.weight.detach()

    def forward(self, input: Tensor) -> Tensor:
        KORNIA_CHECK_SHAPE(input, ["B", "1", "H", "W"])

        B, CH, W, H = input.size()
        self.bin_pooling_kernel = self.bin_pooling_kernel.to(input.dtype).to(input.device)
        self.PoolingConv = self.PoolingConv.to(input.dtype).to(input.device)
        grads = spatial_gradient(input, "diff")
        # unpack the edges
        gx = grads[:, :, 0]
        gy = grads[:, :, 1]
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori = torch.atan2(gy, gx + self.eps) + 2.0 * pi
        o_big = float(self.num_ang_bins) * ori / (2.0 * pi)

        bo0_big_ = torch.floor(o_big)
        wo1_big_ = o_big - bo0_big_
        bo0_big = bo0_big_ % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big_) * mag
        wo1_big = wo1_big_ * mag
        ang_bins = concatenate(
            [
                self.bin_pooling_kernel(
                    (bo0_big == i).to(input.dtype) * wo0_big + (bo1_big == i).to(input.dtype) * wo1_big
                )
                for i in range(0, self.num_ang_bins)
            ],
            1,
        )

        out_no_norm = self.PoolingConv(ang_bins)
        out = normalize(out_no_norm, dim=1, p=2).clamp_(0, float(self.clipval))
        out = normalize(out, dim=1, p=2)
        if self.rootsift:
            out = torch.sqrt(normalize(out, p=1) + self.eps)
        return out