def assert_almost_equal(actual, expected, tolerance=1e-5):
    assert abs(actual - expected) < tolerance, f"Expected {expected}, but got {actual}."