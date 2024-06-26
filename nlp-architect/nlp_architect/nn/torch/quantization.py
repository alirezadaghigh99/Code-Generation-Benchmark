class FakeLinearQuantizationWithSTE(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""

    @staticmethod
    def forward(ctx, input, scale, bits=8):
        """fake quantize input according to scale and number of bits, dequantize
        quantize(input))"""
        return dequantize(quantize(input, scale, bits), scale)

    @staticmethod
    def backward(ctx, grad_output):
        """Calculate estimated gradients for fake quantization using
        Straight-Through Estimator (STE) according to:
        https://openreview.net/pdf?id=B1ae1lZRb"""
        return grad_output, None, None