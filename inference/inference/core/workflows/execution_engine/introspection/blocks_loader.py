def describe_available_blocks() -> BlocksDescription:
    blocks = load_workflow_blocks()
    declared_kinds = []
    result = []
    for block in blocks:
        block_schema = block.manifest_class.model_json_schema()
        outputs_manifest = block.manifest_class.describe_outputs()
        schema_selectors = retrieve_selectors_from_schema(schema=block_schema)
        block_kinds = [
            k
            for s in schema_selectors.values()
            for r in s.allowed_references
            for k in r.kind
        ]
        declared_kinds.extend(block_kinds)
        for output in outputs_manifest:
            declared_kinds.extend(output.kind)
        manifest_type_identifiers = get_manifest_type_identifiers(
            block_schema=block_schema,
            block_source=block.block_source,
            block_identifier=block.identifier,
        )
        result.append(
            BlockDescription(
                manifest_class=block.manifest_class,
                block_class=block.block_class,
                block_schema=block_schema,
                outputs_manifest=outputs_manifest,
                block_source=block.block_source,
                fully_qualified_block_class_name=block.identifier,
                human_friendly_block_name=build_human_friendly_block_name(
                    fully_qualified_name=block.identifier,
                ),
                manifest_type_identifier=manifest_type_identifiers[0],
                manifest_type_identifier_aliases=manifest_type_identifiers[1:],
            )
        )
    _validate_loaded_blocks_names_uniqueness(blocks=result)
    _validate_loaded_blocks_manifest_type_identifiers(blocks=result)
    declared_kinds = list(set(declared_kinds))
    _validate_used_kinds_uniqueness(declared_kinds=declared_kinds)
    return BlocksDescription(blocks=result, declared_kinds=declared_kinds)