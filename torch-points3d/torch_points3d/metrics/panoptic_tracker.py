class _Instance(NamedTuple):
    classname: str
    score: float
    indices: np.array  # type: ignore
    scan_id: int

    def iou(self, other: "_Instance") -> float:
        assert self.scan_id == other.scan_id
        intersection = float(len(np.intersect1d(other.indices, self.indices)))
        return intersection / float(len(other.indices) + len(self.indices) - intersection)

    def find_best_match(self, others: List["_Instance"]) -> Tuple[float, int]:
        ioumax = -np.inf
        best_match = -1
        for i, other in enumerate(others):
            iou = self.iou(other)
            if iou > ioumax:
                ioumax = iou
                best_match = i
        return ioumax, best_match

