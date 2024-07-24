class LastFrameWriterPreprocess(BasicWriterPreprocess):
    """Data writer that writes the last channel of observation.

    This class is designed to be used with ``FrameStackTransitionPicker``.
    """

    def process_observation(self, observation: Observation) -> Observation:
        if isinstance(observation, (list, tuple)):
            return [np.expand_dims(obs[-1], axis=0) for obs in observation]
        else:
            return np.expand_dims(observation[-1], axis=0)

