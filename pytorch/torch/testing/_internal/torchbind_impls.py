def _empty_tensor_queue() -> torch.ScriptObject:
    return torch.classes._TorchScriptTesting._TensorQueue(
        torch.empty(
            0,
        ).fill_(-1)
    )

