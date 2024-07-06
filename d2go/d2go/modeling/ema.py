def FromModel(cls, model: torch.nn.Module, device: str = "", **kwargs):
        ret = cls(**kwargs)
        ret.save_from(model, device)
        return ret

