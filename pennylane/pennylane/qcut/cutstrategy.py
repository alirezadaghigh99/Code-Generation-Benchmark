def _infer_imbalance(k, wire_depths, free_wires, free_gates, imbalance_tolerance=None) -> float:
        """Helper function for determining best imbalance limit."""
        num_wires = len(wire_depths)
        num_gates = sum(wire_depths.values())

        avg_fragment_wires = (num_wires - 1) // k + 1
        avg_fragment_gates = (num_gates - 1) // k + 1
        if free_wires < avg_fragment_wires:
            raise ValueError(
                "`free_wires` should be no less than the average number of wires per fragment. "
                f"Got {free_wires} >= {avg_fragment_wires} ."
            )
        if free_gates < avg_fragment_gates:
            raise ValueError(
                "`free_gates` should be no less than the average number of gates per fragment. "
                f"Got {free_gates} >= {avg_fragment_gates} ."
            )
        if free_gates > num_gates - k:
            # Case where gate depth not limited (`-k` since each fragments has to have >= 1 gates):
            free_gates = num_gates
            # A small adjustment is added to the imbalance factor to prevents small ks from resulting
            # in extremely unbalanced fragments. It will heuristically force the smallest fragment size
            # to be >= 3 if the average fragment size is greater than 5. In other words, tiny fragments
            # are only allowed when average fragmeng size is small in the first place.
            balancing_adjustment = 2 if avg_fragment_gates > 5 else 0
            free_gates = free_gates - (k - 1 + balancing_adjustment)

        depth_imbalance = max(wire_depths.values()) * num_wires / num_gates - 1
        max_imbalance = free_gates / avg_fragment_gates - 1
        imbalance = min(depth_imbalance, max_imbalance)
        if imbalance_tolerance is not None:
            imbalance = min(imbalance, imbalance_tolerance)

        return imbalance

