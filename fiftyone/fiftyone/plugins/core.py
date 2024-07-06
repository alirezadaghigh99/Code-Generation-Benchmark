def find_plugin(name):
    """Returns the path to the plugin on local disk.

    Args:
        name: the plugin name

    Returns:
        the path to the plugin directory
    """
    plugin = _get_plugin(name)
    return plugin.path

def list_downloaded_plugins():
    """Returns a list of all downloaded plugin names.

    Returns:
        a list of plugin names
    """
    return _list_plugins_by_name()

