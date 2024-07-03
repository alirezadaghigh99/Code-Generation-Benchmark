class DiscreteActionValue(ActionValue):
    """Q-function output for discrete action space.

    Args:
        q_values (torch.Tensor):
            Array of Q values whose shape is (batchsize, n_actions)
    """

    def __init__(self, q_values, q_values_formatter=lambda x: x):
        assert isinstance(q_values, torch.Tensor)
        self.device = q_values.device
        self.q_values = q_values
        self.n_actions = q_values.shape[1]
        self.q_values_formatter = q_values_formatter

    @lazy_property
    def greedy_actions(self):
        return self.q_values.detach().argmax(axis=1).int()

    @lazy_property
    def max(self):
        index = self.greedy_actions.long().unsqueeze(1)
        return self.q_values.gather(dim=1, index=index).flatten()

    def evaluate_actions(self, actions):
        index = actions.long().unsqueeze(1)
        return self.q_values.gather(dim=1, index=index).flatten()

    def compute_advantage(self, actions):
        return self.evaluate_actions(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return self.evaluate_actions(actions) - self.evaluate_actions(argmax_actions)

    def compute_expectation(self, beta):
        return torch.sum(F.softmax(beta * self.q_values) * self.q_values, dim=1)

    def __repr__(self):
        return "DiscreteActionValue greedy_actions:{} q_values:{}".format(
            self.greedy_actions.detach().cpu().numpy(),
            self.q_values_formatter(self.q_values.detach().cpu().numpy()),
        )

    @property
    def params(self):
        return (self.q_values,)

    def __getitem__(self, i):
        return DiscreteActionValue(
            self.q_values[i], q_values_formatter=self.q_values_formatter
        )