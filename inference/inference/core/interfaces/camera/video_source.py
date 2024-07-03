class CV2VideoFrameProducer(VideoFrameProducer):
    def __init__(self, video: Union[str, int]):
        self.stream = cv2.VideoCapture(video)

    def isOpened(self) -> bool:
        return self.stream.isOpened()

    def grab(self) -> bool:
        return self.stream.grab()

    def retrieve(self) -> Tuple[bool, ndarray]:
        return self.stream.retrieve()

    def initialize_source_properties(self, properties: Dict[str, float]) -> None:
        for property_id, value in properties.items():
            cv2_id = getattr(cv2, "CAP_PROP_" + property_id.upper())
            self.stream.set(cv2_id, value)

    def discover_source_properties(self) -> SourceProperties:
        width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        return SourceProperties(
            width=width,
            height=height,
            total_frames=total_frames,
            is_file=total_frames > 0,
            fps=fps,
        )

    def release(self):
        self.stream.release()