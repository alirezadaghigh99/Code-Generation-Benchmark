def conv2d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        r = torch.nn.functional.conv2d(x, w)
        if toChannelsLast:
            r = r.contiguous(memory_format=torch.channels_last)
        else:
            r = r.contiguous()
        return r

