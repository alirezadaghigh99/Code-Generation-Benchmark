class SampleConfig(Dict):
    @classmethod
    def from_json(cls, path) -> "SampleConfig":
        file_path = Path(path).resolve()
        with safe_open(file_path) as f:
            loaded_json = json.load(f)
        return cls(loaded_json)

    def update_from_args(self, args, argparser=None):
        if argparser is not None:
            if isinstance(argparser, CustomArgumentParser):
                default_args = {arg for arg in vars(args) if arg not in argparser.seen_actions}
            else:
                # this will fail if we explicitly provide default argument in CLI
                known_args = argparser.parse_known_args()
                default_args = {k for k, v in vars(args).items() if known_args[k] == v}
        else:
            default_args = {k for k, v in vars(args).items() if v is None}

        for key, value in vars(args).items():
            if key not in default_args or key not in self:
                self[key] = value

    def update_from_env(self, key_to_env_dict=None):
        if key_to_env_dict is None:
            key_to_env_dict = _DEFAULT_KEY_TO_ENV
        for k, v in key_to_env_dict:
            if v in os.environ:
                self[k] = int(os.environ[v])

