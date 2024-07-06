def process_non_negative_range(value: ScaleType | None) -> tuple[float, float]:
    result = to_tuple(value if value is not None else 0, 0)
    if not all(x >= 0 for x in result):
        msg = "All values in the non negative range should be non negative"
        raise ValueError(msg)
    return result

def check_1plus(value: tuple[NumericType, NumericType]) -> tuple[NumericType, NumericType]:
    if any(x < 1 for x in value):
        raise ValueError(f"All values should be >= 1, got {value} instead")
    return value

def check_01(value: tuple[NumericType, NumericType]) -> tuple[NumericType, NumericType]:
    if not all(0 <= x <= 1 for x in value):
        raise ValueError(f"All values should be in [0, 1], got {value} instead")
    return value

