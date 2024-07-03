    def from_tensor(cls, t: Optional[_TensorMetadata]) -> Optional["TensorKey"]:
        if t is not None:
            return cls._make(t.id, t.storage_data_ptr, t.allocation_id, t.device)
        return None