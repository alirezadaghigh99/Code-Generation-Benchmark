    def compute_categorical_labels(preds: np.ndarray, truths: np.ndarray) -> np.ndarray:
        preds = np.clip(preds, 0.001, 0.999)  # avoid inf
        labels = entropy(truths, preds, axis=1)
        return labels