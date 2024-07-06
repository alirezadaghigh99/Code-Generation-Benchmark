def get_tensor_size_bytes_map(
    obj: object,
) -> Dict[torch.Tensor, int]:
    tensor_map = {}
    attributes_q = deque()
    attributes_q.append(obj)
    while attributes_q:
        attribute = attributes_q.popleft()
        if isinstance(attribute, torch.Tensor):
            tensor_map[attribute] = attribute.size().numel() * attribute.element_size()
        elif _is_named_tuple(attribute):
            attributes_q.extend(attribute._asdict().values())
        elif isinstance(attribute, Mapping):
            attributes_q.extend(attribute.values())
        elif isinstance(attribute, Sequence) and not isinstance(attribute, str):
            attributes_q.extend(attribute)
        elif hasattr(attribute, "__dict__") and not isinstance(attribute, Enum):
            attributes_q.extend(attribute.__dict__.values())
    return tensor_map

