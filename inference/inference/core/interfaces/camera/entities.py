class VideoFrame:
    """Represents a single frame of video data.

    Attributes:
        image (np.ndarray): The image data of the frame as a NumPy array.
        frame_id (FrameID): A unique identifier for the frame.
        frame_timestamp (FrameTimestamp): The timestamp when the frame was captured.
        source_id (int): The index of the video_reference element which was passed to InferencePipeline for this frame (useful when multiple streams are passed to InferencePipeline).
    """

    image: np.ndarray
    frame_id: FrameID
    frame_timestamp: FrameTimestamp
    source_id: Optional[int] = None

class SourceProperties:
    width: int
    height: int
    total_frames: int
    is_file: bool
    fps: float

