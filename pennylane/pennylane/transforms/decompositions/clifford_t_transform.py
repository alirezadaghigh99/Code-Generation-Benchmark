def pauli_group(x):
        return [qml.Identity(x), qml.X(x), qml.Y(x), qml.Z(x)]

