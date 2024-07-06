def generator():
                return np.eye(2)

def compute_matrix(theta):  # pylint:disable=arguments-differ
        return np.tensordot(theta, np.array([[0.4, 1.2], [1.2, 0.4]]), axes=0)

def matrix(self, _=None):
                return np.eye(2)

def compute_matrix(theta):  # pylint:disable=arguments-differ
        return np.tensordot(theta, np.array([[0.4, 1.2], [1.2, 0.4]]), axes=0)

def adjoint(self):
            return FlipAndRotate(
                -self.parameters[0],
                self.wires[0],
                self.wires[1],
                do_flip=self.hyperparameters["do_flip"],
            )

def compute_decomposition(angle, wires, do_flip):  # pylint: disable=arguments-differ
            op_list = []
            if do_flip:
                op_list.append(qml.PauliX(wires=wires[1]))
            op_list.append(qml.RX(angle, wires=wires[0]))
            return op_list

def compute_decomposition(angle, wires, do_flip):  # pylint: disable=arguments-differ
            op_list = []
            if do_flip:
                op_list.append(qml.PauliX(wires=wires[1]))
            op_list.append(qml.RX(angle, wires=wires[0]))
            return op_list

