def get_basic_sparsity_config(
    input_sample_size=None, sparsity_init=0.02, sparsity_target=0.5, sparsity_target_epoch=2, sparsity_freeze_epoch=3
):
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]

    config = NNCFConfig()
    config.update(
        {
            "model": "basic_sparse_conv",
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {
                "algorithm": "rb_sparsity",
                "sparsity_init": sparsity_init,
                "params": {
                    "schedule": "polynomial",
                    "sparsity_target": sparsity_target,
                    "sparsity_target_epoch": sparsity_target_epoch,
                    "sparsity_freeze_epoch": sparsity_freeze_epoch,
                },
            },
        }
    )
    return config

