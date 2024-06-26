class AtariCategoricalActor(CategoricalActor, AtariBaseModel):
    def __init__(self, state_shape, action_dim, units=None,
                 name="AtariCategoricalActor"):
        self.action_dim = action_dim

        # Build base layers
        AtariBaseModel.__init__(self, name, state_shape)

        # Build top layer
        self.out_prob = Dense(action_dim, activation='softmax')

        CategoricalActor.call(self, tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.uint8)))

    def _compute_features(self, states):
        # Extract features on base layers
        return AtariBaseModel.call(self, states)