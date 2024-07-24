class JPEG(Transform):
    """Apply JPEG compression and decompression to the given images.

    If the input is a :class:`torch.Tensor`, it is expected
    to be of dtype uint8, on CPU, and have [..., 3 or 1, H, W] shape,
    where ... means an arbitrary number of leading dimensions.

    Args:
        quality (sequence or number): JPEG quality, from 1 to 100. Lower means more compression.
            If quality is a sequence like (min, max), it specifies the range of JPEG quality to
            randomly select from (inclusive of both ends).

    Returns:
        image with JPEG compression.
    """

    def __init__(self, quality: Union[int, Sequence[int]]):
        super().__init__()
        if isinstance(quality, int):
            quality = [quality, quality]
        else:
            _check_sequence_input(quality, "quality", req_sizes=(2,))

        if not (1 <= quality[0] <= quality[1] <= 100 and isinstance(quality[0], int) and isinstance(quality[1], int)):
            raise ValueError(f"quality must be an integer from 1 to 100, got {quality =}")

        self.quality = quality

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        quality = torch.randint(self.quality[0], self.quality[1] + 1, ()).item()
        return dict(quality=quality)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.jpeg, inpt, quality=params["quality"])

