def get_transformer_logs(
    attentions: List[torch.Tensor], model: nn.Module, attn_mask: torch.Tensor
):
    logs = {}
    n = attn_mask.sum()
    model_attention_entropy = -sum(
        map(
            lambda x: ((x * torch.log(x + 1e-7)).sum(dim=-1) * attn_mask.unsqueeze(1))
            .sum()
            .item(),
            attentions,
        )
    ) / (len(attentions) * n)
    model_parameter_norm = parameter_norm(model)
    logs["attention_entropy"] = (model_attention_entropy, n * len(attentions))
    logs["parameter_norm"] = (model_parameter_norm, 1)
    return logs