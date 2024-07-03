    def save_to_memory(self, *args, is_vectorised=False):
        """Applies appropriate save_to_memory function depending on whether
        the environment is vectorised or not.

        :param *args: Variable length argument list. Contains batched or unbatched transition elements in consistent order,
            e.g. states, actions, rewards, next_states, dones
        :param is_vectorised: Boolean flag indicating if the environment has been vectorised
        :type is_vectorised: bool
        """
        if is_vectorised:
            self.save_to_memory_vect_envs(*args)
        else:
            self.save_to_memory_single_env(*args)