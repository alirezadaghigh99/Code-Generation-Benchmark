def state(self):
        """Returns the state vector of the circuit prior to measurement.

        .. note::

            Only state vector simulators support this property. Please see the
            plugin documentation for more details.
        """
        raise NotImplementedError

def _get_batch_size(tensor, expected_shape, expected_size):
        """Determine whether a tensor has an additional batch dimension for broadcasting,
        compared to an expected_shape. As QubitDevice does not natively support broadcasting,
        it always reports no batch size, that is ``batch_size=None``"""
        # pylint: disable=unused-argument
        return None

