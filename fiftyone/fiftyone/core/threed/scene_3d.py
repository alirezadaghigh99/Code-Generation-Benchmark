    def from_fo3d(path: str):
        """Loads a scene from an FO3D file.

        Args:
            path: the path to an ``.fo3d`` file

        Returns:
            a :class:`Scene`
        """
        if not path.endswith(".fo3d"):
            raise ValueError("Scene must be loaded from a .fo3d file")

        dict_data = fos.read_json(path)

        return Scene._from_fo3d_dict(dict_data)    def from_fo3d(path: str):
        """Loads a scene from an FO3D file.

        Args:
            path: the path to an ``.fo3d`` file

        Returns:
            a :class:`Scene`
        """
        if not path.endswith(".fo3d"):
            raise ValueError("Scene must be loaded from a .fo3d file")

        dict_data = fos.read_json(path)

        return Scene._from_fo3d_dict(dict_data)