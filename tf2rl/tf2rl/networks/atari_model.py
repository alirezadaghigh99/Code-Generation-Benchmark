class AtariQFunc(AtariBaseModel):
    def __init__(self, state_shape, action_dim, units=None,
                 name="QFunc", enable_dueling_dqn=False,
                 enable_noisy_dqn=False, enable_categorical_dqn=False,
                 n_atoms=51):
        self._enable_dueling_dqn = enable_dueling_dqn
        self._enable_noisy_dqn = enable_noisy_dqn
        self._enable_categorical_dqn = enable_categorical_dqn
        if enable_categorical_dqn:
            self._action_dim = action_dim
            self._n_atoms = n_atoms
            action_dim = (action_dim + int(enable_dueling_dqn)) * n_atoms

        # Build base layers
        super().__init__(name, enable_noisy_dqn)
        DenseLayer = NoisyDense if enable_noisy_dqn else Dense

        self.fc2 = DenseLayer(action_dim, activation='linear')

        if self._enable_dueling_dqn and not enable_categorical_dqn:
            self.fc3 = DenseLayer(1, activation='linear')

        input_shape = (1,) + state_shape
        with tf.device("/cpu:0"):
            self(inputs=tf.constant(np.zeros(shape=input_shape,
                                             dtype=np.float32)))

    def call(self, inputs):
        # Extract features on base layers
        features = super().call(inputs)

        if self._enable_categorical_dqn:
            features = self.fc2(features)
            if self._enable_dueling_dqn:
                features = tf.reshape(
                    features, (-1, self._action_dim + 1, self._n_atoms))  # [batch_size, action_dim, n_atoms]
                v_values = tf.reshape(
                    features[:, 0], (-1, 1, self._n_atoms))
                advantages = tf.reshape(
                    features[:, 1:], [-1, self._action_dim, self._n_atoms])
                features = v_values + (advantages - tf.expand_dims(
                    tf.reduce_mean(advantages, axis=1), axis=1))
            else:
                features = tf.reshape(
                    features, (-1, self._action_dim, self._n_atoms))  # [batch_size, action_dim, n_atoms]
            # [batch_size, action_dim, n_atoms]
            q_dist = tf.keras.activations.softmax(features, axis=2)
            return tf.clip_by_value(q_dist, 1e-8, 1.0 - 1e-8)
        else:
            if self._enable_dueling_dqn:
                advantages = self.fc2(features)
                v_values = self.fc3(features)
                q_values = (v_values
                            + (advantages
                               - tf.reduce_mean(advantages, axis=1, keepdims=True)))
            else:
                q_values = self.fc2(features)
            return q_values

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

