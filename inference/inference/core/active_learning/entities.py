class SamplingMethod:
    name: str
    sample: Callable[[np.ndarray, Prediction, PredictionType], bool]