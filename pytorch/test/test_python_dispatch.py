def backward(ctx, grad_output):
                assert isinstance(grad_output, LoggingTensor)
                (x,) = ctx.saved_tensors
                assert isinstance(x, LoggingTensor)
                escape[0] = x
                return grad_output * 2 * x

