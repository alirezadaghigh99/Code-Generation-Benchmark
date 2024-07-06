def reset() -> None:
    """Clear all compile caches and restore initial state"""
    with convert_frame.compile_lock:
        reset_code_caches()
        convert_frame.input_codes.clear()
        convert_frame.output_codes.clear()
        orig_code_map.clear()
        guard_failures.clear()
        graph_break_reasons.clear()
        resume_execution.ContinueExecutionCache.cache.clear()
        _reset_guarded_backend_cache()
        reset_frame_count()
        torch._C._dynamo.compiled_autograd.clear_cache()
        convert_frame.FRAME_COUNTER = 0
        convert_frame.FRAME_COMPILE_COUNTER.clear()
        callback_handler.clear()
        GenerationTracker.clear()
        torch._dynamo.utils.warn_once_cache.clear()
        torch._C._autograd._saved_tensors_hooks_set_tracing(False)

