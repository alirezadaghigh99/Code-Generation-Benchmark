def retrieve_step_output(
    selector: str,
    execution_cache: ExecutionCache,
    accepts_batch_input: bool,
    step_name: str,
) -> Any:
    value = execution_cache.get_output(selector=selector)
    if not execution_cache.output_represent_batch(selector=selector):
        value = value[0]
    if accepts_batch_input:
        return value
    if isinstance(value, list):
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
    return value