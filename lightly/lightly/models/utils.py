def get_weight_decay_parameters(
    modules: Iterable[Module],
    decay_norm: bool = False,
    decay_bias: bool = False,
    norm_layers: Tuple[Type[Module], ...] = _NORM_LAYERS,
) -> Tuple[List[Parameter], List[Parameter]]:
    """Returns all parameters of the modules that should be decayed and not decayed.

    Args:
        modules:
            List of modules to get the parameters from.
        decay_norm:
            If True, normalization parameters are decayed.
        decay_bias:
            If True, bias parameters are decayed.
        norm_layers:
            Tuple of normalization classes to decay if decay_norm is True.

    Returns:
        (params, params_no_weight_decay) tuple.
    """
    params = []
    params_no_weight_decay = []
    for module in modules:
        for mod in module.modules():
            if isinstance(mod, norm_layers):
                if not decay_norm:
                    params_no_weight_decay.extend(mod.parameters(recurse=False))
                else:
                    params.extend(mod.parameters(recurse=False))
            else:
                for name, param in mod.named_parameters(recurse=False):
                    if not decay_bias and name.endswith("bias"):
                        params_no_weight_decay.append(param)
                    else:
                        params.append(param)
    return params, params_no_weight_decay

def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Converts a batch of input images into patches.

    Args:
        images:
            Images tensor with shape (batch_size, channels, height, width)
        patch_size:
            Patch size in pixels. Image width and height must be multiples of
            the patch size.

    Returns:
        Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2)
        where num_patches = image_width / patch_size * image_height / patch_size.

    """
    # N, C, H, W = (batch_size, channels, height, width)
    N, C, H, W = images.shape
    assert H == W and H % patch_size == 0

    patch_h = patch_w = H // patch_size
    num_patches = patch_h * patch_w
    patches = images.reshape(shape=(N, C, patch_h, patch_size, patch_w, patch_size))
    patches = torch.einsum("nchpwq->nhwpqc", patches)
    patches = patches.reshape(shape=(N, num_patches, patch_size**2 * C))
    return patches

