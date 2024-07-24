class ResBlock(torch.nn.Module):
    """
    Basic ResNet type block

    Parameters
    ----------
    input_nc:
        Number of input channels
    output_nc:
        number of output channels
    convolution
        Either MinkowskConvolution or MinkowskiConvolutionTranspose
    dimension:
        Dimension of the spatial grid
    """

    def __init__(self, input_nc, output_nc, convolution):
        super().__init__()
        self.block = (
            Seq()
            .append(convolution(input_nc, output_nc, kernel_size=3, stride=1))
            .append(snn.BatchNorm(output_nc))
            .append(snn.ReLU())
            .append(convolution(output_nc, output_nc, kernel_size=3, stride=1))
            .append(snn.BatchNorm(output_nc))
            .append(snn.ReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(snn.Conv3d(input_nc, output_nc, kernel_size=1, stride=1)).append(snn.BatchNorm(output_nc))
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out = out + self.downsample(x)
        else:
            out = out + x
        return out

