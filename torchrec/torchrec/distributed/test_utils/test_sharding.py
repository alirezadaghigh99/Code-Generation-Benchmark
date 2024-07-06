def create_test_sharder(
    sharder_type: str,
    sharding_type: str,
    kernel_type: str,
    fused_params: Optional[Dict[str, Any]] = None,
    qcomms_config: Optional[QCommsConfig] = None,
    device: Optional[torch.device] = None,
) -> Union[TestEBSharder, TestEBCSharder, TestETSharder, TestETCSharder]:
    if fused_params is None:
        fused_params = {}
    qcomm_codecs_registry = {}
    if qcomms_config is not None:
        qcomm_codecs_registry = get_qcomm_codecs_registry(qcomms_config, device=device)
    if "learning_rate" not in fused_params:
        fused_params["learning_rate"] = 0.1
    if sharder_type == SharderType.EMBEDDING_BAG.value:
        return TestEBSharder(
            sharding_type, kernel_type, fused_params, qcomm_codecs_registry
        )
    elif sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value:
        return TestEBCSharder(
            sharding_type,
            kernel_type,
            fused_params,
            qcomm_codecs_registry,
        )
    elif sharder_type == SharderType.EMBEDDING_TOWER.value:
        return TestETSharder(
            sharding_type, kernel_type, fused_params, qcomm_codecs_registry
        )
    elif sharder_type == SharderType.EMBEDDING_TOWER_COLLECTION.value:
        return TestETCSharder(
            sharding_type, kernel_type, fused_params, qcomm_codecs_registry
        )
    else:
        raise ValueError(f"Sharder not supported {sharder_type}")

def copy_state_dict(
    loc: Dict[str, Union[torch.Tensor, ShardedTensor, DTensor]],
    glob: Dict[str, torch.Tensor],
    exclude_predfix: Optional[str] = None,
) -> None:
    for name, tensor in loc.items():
        if exclude_predfix is not None and name.startswith(exclude_predfix):
            continue
        else:
            assert name in glob, name
        global_tensor = glob[name]
        if isinstance(global_tensor, ShardedTensor):
            global_tensor = global_tensor.local_shards()[0].tensor
        if isinstance(global_tensor, DTensor):
            # pyre-ignore[16]
            global_tensor = global_tensor.to_local().local_shards()[0]

        if isinstance(tensor, ShardedTensor):
            for local_shard in tensor.local_shards():
                assert global_tensor.ndim == local_shard.tensor.ndim
                shard_meta = local_shard.metadata
                t = global_tensor.detach()
                if t.ndim == 1:
                    t = t[
                        shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                        + local_shard.tensor.shape[0]
                    ]
                elif t.ndim == 2:
                    t = t[
                        shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                        + local_shard.tensor.shape[0],
                        shard_meta.shard_offsets[1] : shard_meta.shard_offsets[1]
                        + local_shard.tensor.shape[1],
                    ]
                else:
                    raise ValueError("Tensors with ndim > 2 are not supported")
                local_shard.tensor.copy_(t)
        elif isinstance(tensor, DTensor):
            shard_offsets = tensor.to_local().local_offsets()  # pyre-ignore[16]
            for i, local_shard in enumerate(tensor.to_local().local_shards()):
                assert global_tensor.ndim == local_shard.ndim
                t = global_tensor.detach()
                local_shape = local_shard.shape
                global_offset = shard_offsets[i]
                if t.ndim == 1:
                    t = t[global_offset[0] : global_offset[0] + local_shape[0]]
                elif t.ndim == 2:
                    t = t[
                        global_offset[0] : global_offset[0] + local_shape[0],
                        global_offset[1] : global_offset[1] + local_shape[1],
                    ]
                else:
                    raise ValueError("Tensors with ndim > 2 are not supported")
                local_shard.copy_(t)
        else:
            tensor.copy_(global_tensor)

