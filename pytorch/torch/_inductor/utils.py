def run_and_get_code(fn, *args, **kwargs):
    from .graph import GraphLowering

    source_codes: List[str] = []

    def save_output_code(code: str):
        source_codes.append(code)

    with mock.patch.object(GraphLowering, "save_output_code", save_output_code):
        torch._dynamo.reset()
        result = fn(*args, **kwargs)
    return result, source_codes

