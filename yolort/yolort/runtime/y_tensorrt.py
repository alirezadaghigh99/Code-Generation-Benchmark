    def predict(self, x: Any, image_loader: Optional[Callable] = None) -> List[Dict[str, Tensor]]:
        """
        Predict function for raw data or processed data
        Args:
            x: Input to predict. Can be raw data or processed data.
            image_loader: Utility function to convert raw data to Tensor.

        Returns:
            The post-processed model predictions.
        """
        image_loader = image_loader or self.default_loader
        images = self.collate_images(x, image_loader)
        return self.forward(images)