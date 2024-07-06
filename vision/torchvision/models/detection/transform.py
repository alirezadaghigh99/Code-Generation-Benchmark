def resize(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            if self._skip_resize:
                return image, target
            size = self.torch_choice(self.min_size)
        else:
            size = self.min_size[-1]
        image, target = _resize_image_and_masks(image, size, self.max_size, target, self.fixed_size)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target

