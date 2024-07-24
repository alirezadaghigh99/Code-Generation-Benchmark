class RGB2YUVBT601(aug.Augmentation):
    def __init__(self):
        super().__init__()
        self.trans = InvertibleColorTransform(
            convert_rgb_to_yuv_bt601, convery_yuv_bt601_to_rgb
        )

    def get_transform(self, image) -> Transform:
        return self.trans

class YUVBT6012RGB(aug.Augmentation):
    def __init__(self):
        super().__init__()
        self.trans = InvertibleColorTransform(
            convery_yuv_bt601_to_rgb, convert_rgb_to_yuv_bt601
        )

    def get_transform(self, image) -> Transform:
        return self.trans

