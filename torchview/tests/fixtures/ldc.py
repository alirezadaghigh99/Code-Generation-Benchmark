class LDC(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self) -> None:
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2,)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(2, 32, 64)  # [128,256,100,100]
        self.dblock_4 = _DenseBlock(3, 64, 96)  # 128
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.side_2 = SingleConvBlock(32, 64, 2)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(32, 64, 2)
        self.pre_dense_3 = SingleConvBlock(32, 64, 1)
        self.pre_dense_4 = SingleConvBlock(64, 96, 1)  # 128

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(64, 2)
        self.up_block_4 = UpConvBlock(96, 3)  # 128
        self.block_cat = CoFusion(4, 4)  # cats fusion method

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        assert x.ndim == 4, x.shape
        # supose the image size is 352x352
        # Block 1
        block_1 = self.block_1(x)  # [8,16,176,176]
        block_1_side = self.side_1(block_1)  # 16 [8,32,88,88]

        # Block 2
        block_2 = self.block_2(block_1)  # 32 - [8,32,176,176]
        block_2_down = self.maxpool(block_2)  # [8,32,88,88]
        block_2_add = block_2_down + block_1_side  # [8,32,88,88]
        block_2_side = self.side_2(block_2_add)  # [8,64,44,44] block 3 R connection

        # Block 3
        block_3_pre_dense = self.pre_dense_3(
            block_2_down)  # [8,64,88,88] block 3 L connection
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])  # [8,64,88,88]
        block_3_down = self.maxpool(block_3)  # [8,64,44,44]
        block_3_add = block_3_down + block_2_side  # [8,64,44,44]

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)  # [8,64,44,44]
        block_4_pre_dense = self.pre_dense_4(
            block_3_down+block_2_resize_half)  # [8,96,44,44]
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])  # [8,96,44,44]

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        results: list[torch.Tensor] = [out_1, out_2, out_3, out_4]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        results.append(block_cat)
        return results

