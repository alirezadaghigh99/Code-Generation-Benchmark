def main() -> None:
    parser_data = {
        "download": download_fn,
        "chat": chat_fn,
        "finetune": finetune_lora_fn,
        "finetune_lora": finetune_lora_fn,
        "finetune_full": finetune_full_fn,
        "finetune_adapter": finetune_adapter_fn,
        "finetune_adapter_v2": finetune_adapter_v2_fn,
        "pretrain": pretrain_fn,
        "generate": generate_base_fn,
        "generate_full": generate_full_fn,
        "generate_adapter": generate_adapter_fn,
        "generate_adapter_v2": generate_adapter_v2_fn,
        "generate_sequentially": generate_sequentially_fn,
        "generate_tp": generate_tp_fn,
        "convert_to_litgpt": convert_hf_checkpoint_fn,
        "convert_from_litgpt": convert_lit_checkpoint_fn,
        "convert_pretrained_checkpoint": convert_pretrained_checkpoint_fn,
        "merge_lora": merge_lora_fn,
        "evaluate": evaluate_fn,
        "serve": serve_fn
    }

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    torch.set_float32_matmul_precision("high")
    CLI(parser_data)