def from_json(cls, path: str) -> "NNCFConfig":
        """
        Load NNCF config from a JSON file at `path`.

        :param path: Path to the .json file containing the NNCF configuration.
        """
        file_path = Path(path)
        with safe_open(file_path) as f:
            loaded_json = json.load(f)
        return cls.from_dict(loaded_json)

def from_dict(cls, nncf_dict: Dict) -> "NNCFConfig":
        """
        Load NNCF config from a Python dictionary. The dict must contain only JSON-supported primitives.

        :param nncf_dict: A Python dict with the JSON-style configuration for NNCF.
        """

        NNCFConfig.validate(nncf_dict)
        return cls(deepcopy(nncf_dict))

