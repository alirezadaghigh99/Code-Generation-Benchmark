    def _get_n_step_info(self, n_step_buffer, gamma):
        """Returns n step reward, next_state, and done, as well as other saved transition elements, in order."""
        # info of the last transition
        # t = [n_step_buffer[0]]
        t = [n_step_buffer[0]]
        transition = self._process_transition(t, np_array=True)

        vect_reward = transition["reward"][0]
        vect_next_state = transition["next_state"][0]
        if "done" in transition.keys():
            vect_done = transition["done"][0]
        elif "termination" in transition.keys():
            vect_done = transition["termination"][0]
        else:
            vect_done = transition["terminated"][0]

        for idx, ts in enumerate(list(n_step_buffer)[1:]):
            if not vect_done:
                vect_r, vect_n_s = (ts.reward, ts.next_state)

                if "done" in transition.keys():
                    vect_d = ts.done
                elif "termination" in transition.keys():
                    vect_d = ts.termination
                else:
                    vect_d = ts.terminated

                vect_reward += vect_r * gamma ** (idx + 1)
                vect_done = np.array([vect_d])
                vect_next_state = vect_n_s

        transition["reward"] = vect_reward
        transition["next_state"] = vect_next_state
        if "done" in transition.keys():
            transition["done"] = vect_done
        elif "termination" in transition.keys():
            transition["termination"] = vect_done
        else:
            transition["terminated"] = vect_done
        transition["state"] = transition["state"][0]
        transition["action"] = transition["action"][0]

        return tuple(transition.values())