def tensor_from_numpy(npy_img: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Read an image file as a PyTorch tensor

    Args:
    ----
        npy_img: image encoded as a numpy array of shape (H, W, C) in np.uint8
        dtype: the desired data type of the output tensor. If it is float-related, values will be divided by 255.

    Returns:
    -------
        same image as a tensor of shape (C, H, W)
    """
    if dtype not in (torch.uint8, torch.float16, torch.float32):
        raise ValueError("insupported value for dtype")

    if dtype == torch.float32:
        img = to_tensor(npy_img)
    else:
        img = torch.from_numpy(npy_img)
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        if dtype == torch.float16:
            # Switch to FP16
            img = img.to(dtype=torch.float16).div(255)

    return img