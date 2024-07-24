class Resize(torch.nn.Module):
    def __init__(self, resize_size: Tuple[int, ...], interpolation_type: str = "bilinear") -> None:
        super().__init__()
        self.resize_size = list(resize_size)  # doing this to keep mypy happy
        self._interpolation_mode_strategy = InterpolationStrategy(interpolation_type)

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        resized_images = ()
        resized_disparities = ()
        resized_masks = ()

        INTERP_MODE = self._interpolation_mode_strategy()

        for img in images:
            # We hard-code antialias=False to preserve results after we changed
            # its default from None to True (see
            # https://github.com/pytorch/vision/pull/7160)
            # TODO: we could re-train the stereo models with antialias=True?
            resized_images += (F.resize(img, self.resize_size, interpolation=INTERP_MODE, antialias=False),)

        for dsp in disparities:
            if dsp is not None:
                # rescale disparity to match the new image size
                scale_x = self.resize_size[1] / dsp.shape[-1]
                resized_disparities += (F.resize(dsp, self.resize_size, interpolation=INTERP_MODE) * scale_x,)
            else:
                resized_disparities += (None,)

        for mask in masks:
            if mask is not None:
                resized_masks += (
                    # we squeeze and unsqueeze because the API requires > 3D tensors
                    F.resize(
                        mask.unsqueeze(0),
                        self.resize_size,
                        interpolation=F.InterpolationMode.NEAREST,
                    ).squeeze(0),
                )
            else:
                resized_masks += (None,)

        return resized_images, resized_disparities, resized_masks

class ToTensor(torch.nn.Module):
    def forward(
        self,
        images: Tuple[PIL.Image.Image, PIL.Image.Image],
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        if images[0] is None:
            raise ValueError("img_left is None")
        if images[1] is None:
            raise ValueError("img_right is None")

        img_left = F.pil_to_tensor(images[0])
        img_right = F.pil_to_tensor(images[1])
        disparity_tensors = ()
        mask_tensors = ()

        for idx in range(2):
            disparity_tensors += (torch.from_numpy(disparities[idx]),) if disparities[idx] is not None else (None,)
            mask_tensors += (torch.from_numpy(masks[idx]),) if masks[idx] is not None else (None,)

        return (img_left, img_right), disparity_tensors, mask_tensors

class Resize(torch.nn.Module):
    def __init__(self, resize_size: Tuple[int, ...], interpolation_type: str = "bilinear") -> None:
        super().__init__()
        self.resize_size = list(resize_size)  # doing this to keep mypy happy
        self._interpolation_mode_strategy = InterpolationStrategy(interpolation_type)

    def forward(
        self,
        images: T_STEREO_TENSOR,
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        resized_images = ()
        resized_disparities = ()
        resized_masks = ()

        INTERP_MODE = self._interpolation_mode_strategy()

        for img in images:
            # We hard-code antialias=False to preserve results after we changed
            # its default from None to True (see
            # https://github.com/pytorch/vision/pull/7160)
            # TODO: we could re-train the stereo models with antialias=True?
            resized_images += (F.resize(img, self.resize_size, interpolation=INTERP_MODE, antialias=False),)

        for dsp in disparities:
            if dsp is not None:
                # rescale disparity to match the new image size
                scale_x = self.resize_size[1] / dsp.shape[-1]
                resized_disparities += (F.resize(dsp, self.resize_size, interpolation=INTERP_MODE) * scale_x,)
            else:
                resized_disparities += (None,)

        for mask in masks:
            if mask is not None:
                resized_masks += (
                    # we squeeze and unsqueeze because the API requires > 3D tensors
                    F.resize(
                        mask.unsqueeze(0),
                        self.resize_size,
                        interpolation=F.InterpolationMode.NEAREST,
                    ).squeeze(0),
                )
            else:
                resized_masks += (None,)

        return resized_images, resized_disparities, resized_masks

class ToTensor(torch.nn.Module):
    def forward(
        self,
        images: Tuple[PIL.Image.Image, PIL.Image.Image],
        disparities: Tuple[T_FLOW, T_FLOW],
        masks: Tuple[T_MASK, T_MASK],
    ) -> Tuple[T_STEREO_TENSOR, Tuple[T_FLOW, T_FLOW], Tuple[T_MASK, T_MASK]]:
        if images[0] is None:
            raise ValueError("img_left is None")
        if images[1] is None:
            raise ValueError("img_right is None")

        img_left = F.pil_to_tensor(images[0])
        img_right = F.pil_to_tensor(images[1])
        disparity_tensors = ()
        mask_tensors = ()

        for idx in range(2):
            disparity_tensors += (torch.from_numpy(disparities[idx]),) if disparities[idx] is not None else (None,)
            mask_tensors += (torch.from_numpy(masks[idx]),) if masks[idx] is not None else (None,)

        return (img_left, img_right), disparity_tensors, mask_tensors

