    def reset(self):
        next_state, target = self._new_state_and_target_action()
        next_reward = np.zeros(self.arms)
        next_reward[target] = 1
        self.prev_reward = next_reward
        return next_state