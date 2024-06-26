def retrieve_value_from_runtime_input(
    selector: str,
    runtime_parameters: Dict[str, Any],
    accepts_batch_input: bool,
    step_name: str,
) -> Any:
    try:
        parameter_name = get_last_chunk_of_selector(selector=selector)
        value = runtime_parameters[parameter_name]
        if not _retrieved_inference_image(value=value) or accepts_batch_input:
            return value
        if len(value) > 1:
            raise ExecutionEngineNotImplementedError(
                public_message=f"Step `{step_name}` defines input pointing to {selector} which "
                f"ships batch input of size larger than one, but at the same time workflow block "
                f"used to implement the step does not accept batch input. That may be "
                f"for instance the case for steps with flow-control, as workflows execution engine "
                f"does not yet support branching when control-flow decision is made element-wise.",
                context="workflow_execution | steps_parameters_assembling",
            )
        return value[0]
    except KeyError as e:
        raise ExecutionEngineRuntimeError(
            public_message=f"Attempted to retrieve runtime parameter using selector {selector} "
            f"discovering miss in runtime parameters. This should have been detected "
            f"by execution engine at the earlier stage. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | steps_parameters_assembling",
            inner_error=e,
        ) from e