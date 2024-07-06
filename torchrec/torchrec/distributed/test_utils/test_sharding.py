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

