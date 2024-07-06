def pruning_modifier_serialization_vals_test(
    yaml_modifier,
    serialized_modifier,
    obj_modifier,
    exclude_mask=False,
):
    assert (
        yaml_modifier.init_sparsity
        == serialized_modifier.init_sparsity
        == obj_modifier.init_sparsity
    )
    assert (
        yaml_modifier.final_sparsity
        == serialized_modifier.final_sparsity
        == obj_modifier.final_sparsity
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
    assert yaml_modifier.params == serialized_modifier.params == obj_modifier.params
    assert (
        yaml_modifier.inter_func
        == serialized_modifier.inter_func
        == obj_modifier.inter_func
    )
    assert (
        str(yaml_modifier.mask_type)
        == str(serialized_modifier.mask_type)
        == str(obj_modifier.mask_type)
    )

def sparsity_mask_creator_test(tensor_shapes, mask_creator, sparsity_val, device):
    tensors = [torch.randn(tensor_shape).to(device) for tensor_shape in tensor_shapes]
    update_masks = mask_creator.create_sparsity_masks(tensors, sparsity_val)

    if isinstance(sparsity_val, float):
        sparsity_val = [sparsity_val] * len(update_masks)

    for update_mask, target_sparsity in zip(update_masks, sparsity_val):
        assert abs(tensor_sparsity(update_mask) - target_sparsity) < 1e-2

        if not isinstance(mask_creator, GroupedPruningMaskCreator):
            _test_num_masked(update_mask, target_sparsity)

    if isinstance(mask_creator, GroupedPruningMaskCreator):
        grouped_masks_test(update_masks, mask_creator, sparsity_val)

    return update_masks

