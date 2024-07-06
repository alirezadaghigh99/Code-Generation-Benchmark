def model_tracer():
    """Simple context manager for tracing. Also captures module constructors"""
    with tracer_context():
        with model_constructor_tracer():
            yield True

