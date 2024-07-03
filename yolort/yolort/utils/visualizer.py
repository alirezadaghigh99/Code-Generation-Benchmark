    def draw_instance_predictions(self, predictions: Dict[str, Tensor]):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (dict): the output of an instance detection model. Following
                fields will be used to draw: "boxes", "labels", "scores".

        Returns:
            np.ndarray: image object with visualizations.
        """

        boxes = self._convert_boxes(predictions["boxes"])
        labels = predictions["labels"].tolist()
        colors = self._create_colors(labels)
        scores = predictions["scores"].tolist()
        labels = self._create_text_labels(labels, scores)

        self.overlay_instances(boxes=boxes, labels=labels, colors=colors)
        return self.output