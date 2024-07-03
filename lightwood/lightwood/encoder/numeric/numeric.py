    def get_weights(self, label_data):
        # get a sorted list of intervals to assign weights. Keys are the interval edges.
        target_weight_keys = np.array(list(self.target_weights.keys()))
        target_weight_values = np.array(list(self.target_weights.values()))
        sorted_indices = np.argsort(target_weight_keys)

        # get sorted arrays for vector numpy operations
        target_weight_keys = target_weight_keys[sorted_indices]
        target_weight_values = target_weight_values[sorted_indices]

        # find the indices of the bins according to the keys. clip to the length of the weight values (search sorted
        # returns indices from 0 to N with N = len(target_weight_keys).
        assigned_target_weight_indices = np.clip(a=np.searchsorted(target_weight_keys, label_data),
                                                 a_min=0,
                                                 a_max=len(target_weight_keys) - 1).astype(np.int32)

        return target_weight_values[assigned_target_weight_indices]