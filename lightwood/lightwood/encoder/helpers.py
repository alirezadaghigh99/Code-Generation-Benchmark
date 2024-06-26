    def prepare(self, x: np.ndarray) -> None:
        # @TODO: streamline input type
        if isinstance(x[0], list):
            x = np.vstack(x)
        if isinstance(x[0], torch.Tensor):
            x = torch.stack(x).numpy()
        if len(x.shape) < 2:
            x = np.expand_dims(x, axis=1)

        x[x == None] = 0 # noqa
        x = x.astype(float)
        self.abs_mean = np.mean(np.abs(x))
        self.scaler.fit(x.reshape(x.size, -1))