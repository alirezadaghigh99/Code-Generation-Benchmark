    def assert_equal(old: Optional[ShapeEnv], new: ShapeEnv) -> ShapeEnv:
        if old is not None:
            assert old is new, "call with different ShapeEnv"
        return new