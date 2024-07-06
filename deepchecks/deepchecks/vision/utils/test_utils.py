def un_normalize_batch(tensor: torch.Tensor, mean: Sized, std: Sized, max_pixel_value: int = 255):
    """Apply un-normalization on a tensor in order to display an image."""
    dim = len(mean)
    reshape_shape = (1, 1, 1, dim)
    max_pixel_value = [max_pixel_value] * dim
    mean = torch.tensor(mean, device=tensor.device).reshape(reshape_shape)
    std = torch.tensor(std, device=tensor.device).reshape(reshape_shape)
    tensor = (tensor * std) + mean
    tensor = tensor * torch.tensor(max_pixel_value, device=tensor.device).reshape(reshape_shape)
    return object_to_numpy(tensor)

def replace_collate_fn_visiondata(vision_data: VisionData, new_collate_fn):
    """Create a new VisionData based on the same attributes as the old one with updated collate function."""
    new_data_loader = replace_collate_fn_dataloader(vision_data.batch_loader, new_collate_fn)
    return VisionData(new_data_loader, task_type=vision_data.task_type.value, reshuffle_data=False,
                      dataset_name=vision_data.name,
                      label_map=vision_data.label_map)

def replace_collate_fn_dataloader(data_loader: DataLoader, new_collate_fn):
    """Replace collate_fn function in DataLoader."""
    props = _get_data_loader_props(data_loader)
    _collisions_removal_dataloader_props(props)
    props['collate_fn'] = new_collate_fn
    return data_loader.__class__(**props)

