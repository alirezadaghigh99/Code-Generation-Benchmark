def apply(fn):
            return lambda x, y, z: fn(fn(x, y), z)

