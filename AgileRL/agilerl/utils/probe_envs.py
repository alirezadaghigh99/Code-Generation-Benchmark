class FixedObsPolicyContActionsImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 1.0, (3, 32, 32))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.sample_obs = [np.zeros((1, 3, 32, 32))]
        self.sample_actions = [np.array([[1.0]])]
        self.q_values = np.array([[0.0]])  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table
        self.policy_values = [[1.0]]  # Correct policy to learn

    def step(self, action):
        observation = np.zeros((3, 32, 32))
        reward = -((1 - action[0]) ** 2)  # Reward depends on action
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((3, 32, 32))
        info = {}
        return observation, info