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

class PluginPackage:
    """Plugin package.

    Args:
        name: the name of the plugin
        path: the path to the plugin's root directory
    """

    name: str
    path: str

    def __repr__(self):
        return f"Plugin(name={self.name}, path={self.path})"

