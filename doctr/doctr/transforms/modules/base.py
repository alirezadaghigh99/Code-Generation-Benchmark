class RandomCrop(NestedObject):
    """Randomly crop a tensor image and its boxes

    Args:
    ----
        scale: tuple of floats, relative (min_area, max_area) of the crop
        ratio: tuple of float, relative (min_ratio, max_ratio) where ratio = h/w
    """

    def __init__(self, scale: Tuple[float, float] = (0.08, 1.0), ratio: Tuple[float, float] = (0.75, 1.33)) -> None:
        self.scale = scale
        self.ratio = ratio

    def extra_repr(self) -> str:
        return f"scale={self.scale}, ratio={self.ratio}"

    def __call__(self, img: Any, target: np.ndarray) -> Tuple[Any, np.ndarray]:
        scale = random.uniform(self.scale[0], self.scale[1])
        ratio = random.uniform(self.ratio[0], self.ratio[1])

        height, width = img.shape[:2]

        # Calculate crop size
        crop_area = scale * width * height
        aspect_ratio = ratio * (width / height)
        crop_width = int(round(math.sqrt(crop_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(crop_area / aspect_ratio)))

        # Ensure crop size does not exceed image dimensions
        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)

        # Randomly select crop position
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)

        # relative crop box
        crop_box = (x / width, y / height, (x + crop_width) / width, (y + crop_height) / height)
        if target.shape[1:] == (4, 2):
            min_xy = np.min(target, axis=1)
            max_xy = np.max(target, axis=1)
            _target = np.concatenate((min_xy, max_xy), axis=1)
        else:
            _target = target

        # Crop image and targets
        croped_img, crop_boxes = F.crop_detection(img, _target, crop_box)
        # hard fallback if no box is kept
        if crop_boxes.shape[0] == 0:
            return img, target
        # clip boxes
        return croped_img, np.clip(crop_boxes, 0, 1)