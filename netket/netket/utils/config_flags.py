    def update(self, name, value):
        """
        Updates a configuration variable in netket.

        Args:
            name: the name of the variable
            value: the new value
        """
        name = name.upper()

        if not self._editable_at_runtime[name]:
            raise RuntimeError(
                f"\n\nFlag `{name}` can only be set through an environment "
                "variable before importing netket.\n"
                "Try launching python with:\n\n"
                f"\t{name}={self.FLAGS[name]} python\n\n"
                "or execute the following snippet BEFORE importing netket:\n\n"
                "\t>>>import os\n"
                f'\t>>>os.environ["{name}"]="{self.FLAGS[name]}"\n'
                "\t>>>import netket as nk\n\n"
            )

        if not isinstance(value, self._types[name]):
            raise TypeError(
                f"Configuration {name} must be a {self._types[name]}, but the "
                f"value {value} is a {type(value)}."
            )

        self._values[name] = self._types[name](value)