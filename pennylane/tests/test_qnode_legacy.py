def execute(self, circuits, execution_config=None):
        return (0,)

class PhaseShift(qml.PhaseShift):
            """custom phase shift."""

            grad_method = None

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

