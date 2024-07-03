            def generator():
                return np.eye(2)    def compute_matrix(theta):  # pylint:disable=arguments-differ
        return np.tensordot(theta, np.array([[0.4, 1.2], [1.2, 0.4]]), axes=0)            def matrix(self, _=None):
                return np.eye(2)    def compute_matrix(theta):  # pylint:disable=arguments-differ
        return np.tensordot(theta, np.array([[0.4, 1.2], [1.2, 0.4]]), axes=0)        def adjoint(self):
            return FlipAndRotate(
                -self.parameters[0],
                self.wires[0],
                self.wires[1],
                do_flip=self.hyperparameters["do_flip"],
            )