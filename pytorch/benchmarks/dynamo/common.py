def export(model, example_inputs):
    example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
    example_outputs = model(*example_args, **example_kwargs)
    _register_dataclass_output_as_pytree(example_outputs)

    ep = torch.export.export(model, example_args, example_kwargs)

    def opt_export(_, example_inputs):
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return ep(*example_args, **example_kwargs)

    return opt_export

def cpu(self) -> Self:
        self.onnx_session.set_providers(["CPUExecutionProvider"])
        return self

def export(model, example_inputs):
    example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
    example_outputs = model(*example_args, **example_kwargs)
    _register_dataclass_output_as_pytree(example_outputs)

    ep = torch.export.export(model, example_args, example_kwargs)

    def opt_export(_, example_inputs):
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return ep(*example_args, **example_kwargs)

    return opt_export

