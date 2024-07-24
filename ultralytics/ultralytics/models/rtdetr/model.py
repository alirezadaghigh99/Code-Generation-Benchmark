class RTDETR(Model):
    """
    Interface for Baidu's RT-DETR model. This Vision Transformer-based object detector provides real-time performance
    with high accuracy. It supports efficient hybrid encoding, IoU-aware query selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model. Defaults to 'rtdetr-l.pt'.
    """

    def __init__(self, model="rtdetr-l.pt") -> None:
        """
        Initializes the RT-DETR model with the given pre-trained model file. Supports .pt and .yaml formats.

        Args:
            model (str): Path to the pre-trained model. Defaults to 'rtdetr-l.pt'.

        Raises:
            NotImplementedError: If the model file extension is not 'pt', 'yaml', or 'yml'.
        """
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """
        Returns a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            dict: A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }

