    class RX(qml.RX):
        def generator(self):
            return qml.Hadamard(self.wires)