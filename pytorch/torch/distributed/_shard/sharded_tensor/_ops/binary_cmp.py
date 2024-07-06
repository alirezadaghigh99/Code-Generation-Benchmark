def equal(types, args, kwargs, process_group):
    return binary_cmp(torch.equal, types, args, kwargs, process_group)

