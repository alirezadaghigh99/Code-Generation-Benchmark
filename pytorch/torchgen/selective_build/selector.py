def from_yaml_str(config_contents: str) -> SelectiveBuilder:
        contents = yaml.safe_load(config_contents)
        return SelectiveBuilder.from_yaml_dict(contents)

