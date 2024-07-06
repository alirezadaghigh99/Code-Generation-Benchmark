def split(self, labels, centers, scores):
        """Split the cluster node into two subclusters."""
        self.left = _BisectingTree(
            indices=self.indices[labels == 0], center=centers[0], score=scores[0]
        )
        self.right = _BisectingTree(
            indices=self.indices[labels == 1], center=centers[1], score=scores[1]
        )

        # reset the indices attribute to save memory
        self.indices = None

