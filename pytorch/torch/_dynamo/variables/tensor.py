    def as_tensor(self, tx):
        if self._tensor_var is None:
            from .builder import SourcelessBuilder

            self._tensor_var = SourcelessBuilder.create(
                tx, torch.scalar_tensor
            ).call_function(tx, [self], {})
        return self._tensor_var