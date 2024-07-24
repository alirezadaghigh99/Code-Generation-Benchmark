class KeypointMetadata(NamedTuple):
    names: List[str]
    flip_map: List[Tuple[str, str]]
    connection_rules: List[Tuple[str, str, Tuple[int, int, int]]]

    def to_dict(self):
        return {
            "keypoint_names": self.names,
            "keypoint_flip_map": self.flip_map,
            "keypoint_connection_rules": self.connection_rules,
        }

