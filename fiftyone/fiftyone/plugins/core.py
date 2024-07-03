def find_plugin(name):
    """Returns the path to the plugin on local disk.

    Args:
        name: the plugin name

    Returns:
        the path to the plugin directory
    """
    plugin = _get_plugin(name)
    return plugin.path