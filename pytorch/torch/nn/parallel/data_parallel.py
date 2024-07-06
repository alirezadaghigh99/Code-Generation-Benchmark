def replicate(
        self, module: T, device_ids: Sequence[Union[int, torch.device]]
    ) -> List[T]:
        return replicate(module, device_ids, not torch.is_grad_enabled())

