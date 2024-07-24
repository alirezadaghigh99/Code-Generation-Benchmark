class ConvertImageDtype(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, img1, img2, flow, valid_flow_mask):
        img1 = F.convert_image_dtype(img1, dtype=self.dtype)
        img2 = F.convert_image_dtype(img2, dtype=self.dtype)

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2, flow, valid_flow_mask

class RandomErasing(T.RandomErasing):
    # This only erases img2, and with an extra max_erase param
    # This max_erase is needed because in the RAFT training ref does:
    # 0 erasing with .5 proba
    # 1 erase with .25 proba
    # 2 erase with .25 proba
    # and there's no accurate way to achieve this otherwise.
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, max_erase=1):
        super().__init__(p=p, scale=scale, ratio=ratio, value=value, inplace=inplace)
        self.max_erase = max_erase
        if self.max_erase <= 0:
            raise ValueError("max_raise should be greater than 0")

    def forward(self, img1, img2, flow, valid_flow_mask):
        if torch.rand(1) > self.p:
            return img1, img2, flow, valid_flow_mask

        for _ in range(torch.randint(self.max_erase, size=(1,)).item()):
            x, y, h, w, v = self.get_params(img2, scale=self.scale, ratio=self.ratio, value=[self.value])
            img2 = F.erase(img2, x, y, h, w, v, self.inplace)

        return img1, img2, flow, valid_flow_mask

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img1, img2, flow, valid_flow_mask):
        img1 = F.normalize(img1, mean=self.mean, std=self.std)
        img2 = F.normalize(img2, mean=self.mean, std=self.std)

        return img1, img2, flow, valid_flow_mask

