def override_quantized_engine(qengine):
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = qengine
    try:
        yield
    finally:
        torch.backends.quantized.engine = previous

def qengine_is_qnnpack():
    return torch.backends.quantized.engine == 'qnnpack'
