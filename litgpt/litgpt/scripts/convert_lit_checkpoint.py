def check_conversion_supported(lit_weights: Dict[str, torch.Tensor]) -> None:
    if any("lora" in wn for wn in lit_weights):
        raise ValueError("Checkpoints with LoRA weights cannot be converted. Call `scripts/merge_lora.py` first.")
    if any("adapter" in wn or "gating_factor" in wn for wn in lit_weights):
        raise NotImplementedError("Converting adapter models is not supported.")

