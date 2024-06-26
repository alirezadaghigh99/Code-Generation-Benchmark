class PettingZooVectorizationParallelWrapper(PettingZooAutoResetParallelWrapper):
    def __init__(self, env: ParallelEnv[AgentID, ObsType, ActionType], n_envs: int):
        super().__init__(env=env)
        self.num_envs = n_envs
        self.env = SubprocVecEnv([lambda: self.env for _ in range(n_envs)])
        return