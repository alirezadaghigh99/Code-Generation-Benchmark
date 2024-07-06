def get_manifest_type_identifiers(
    block_schema: dict,
    block_source: str,
    block_identifier: str,
) -> List[str]:
    if "type" not in block_schema["properties"]:
        raise PluginInterfaceError(
            public_message="Required `type` property not defined for block "
            f"`{block_identifier}` loaded from `{block_source}",
            context="workflow_compilation | blocks_loading",
        )
    constant_literal = block_schema["properties"]["type"].get("const")
    if constant_literal is not None:
        return [constant_literal]
    valid_aliases = block_schema["properties"]["type"].get("enum", [])
    if len(valid_aliases) > 0:
        return valid_aliases
    raise PluginInterfaceError(
        public_message="`type` property for block is required to be `Literal` "
        "defining at least one unique value to identify block in JSON "
        f"definitions. Block `{block_identifier}` loaded from `{block_source} "
        f"does not fit that requirement.",
        context="workflow_compilation | blocks_loading",
    )

def load_blocks_from_plugin(plugin_name: str) -> List[BlockSpecification]:
    try:
        return _load_blocks_from_plugin(plugin_name=plugin_name)
    except ImportError as e:
        raise PluginLoadingError(
            public_message=f"It is not possible to load workflow plugin `{plugin_name}`. "
            f"Make sure the library providing custom step is correctly installed in Python environment.",
            context="workflow_compilation | blocks_loading",
            inner_error=e,
        ) from e
    except AttributeError as e:
        raise PluginInterfaceError(
            public_message=f"Provided workflow plugin `{plugin_name}` do not implement blocks loading "
            f"interface correctly and cannot be loaded.",
            context="workflow_compilation | blocks_loading",
            inner_error=e,
        ) from e

def load_initializers_from_plugin(
    plugin_name: str,
) -> Dict[str, Union[Any, Callable[[None], Any]]]:
    try:
        logging.info(f"Loading workflows initializers from plugin {plugin_name}")
        return _load_initializers_from_plugin(plugin_name=plugin_name)
    except ImportError as e:
        raise PluginLoadingError(
            public_message=f"It is not possible to load workflow plugin `{plugin_name}`. "
            f"Make sure the library providing custom step is correctly installed in Python environment.",
            context="workflow_compilation | blocks_loading",
            inner_error=e,
        ) from e

