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