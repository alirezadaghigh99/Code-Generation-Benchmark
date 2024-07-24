class QuadraticActionValue(ActionValue):
    """Q-function output for continuous action space.

    See: http://arxiv.org/abs/1603.00748

    Define a Q(s,a) with A(s,a) in a quadratic form.

    Q(s,a) = V(s,a) + A(s,a)
    A(s,a) = -1/2 (u - mu(s))^T P(s) (u - mu(s))

    Args:
        mu (torch.Tensor): mu(s), actions that maximize A(s,a)
        mat (torch.Tensor): P(s), coefficient matrices of A(s,a).
          It must be positive definite.
        v (torch.Tensor): V(s), values of s
        min_action (ndarray): minimum action, not batched
        max_action (ndarray): maximum action, not batched
    """

    def __init__(self, mu, mat, v, min_action=None, max_action=None):
        self.mu = mu
        self.mat = mat
        self.v = v
        self.device = mu.device

        if isinstance(min_action, (int, float)):
            min_action = [min_action]

        if min_action is None:
            self.min_action = None
        else:
            self.min_action = torch.as_tensor(min_action).to(self.device).float()
        if isinstance(max_action, (int, float)):
            max_action = [max_action]
        if max_action is None:
            self.max_action = None
        else:
            self.max_action = torch.as_tensor(max_action).to(self.device).float()
        self.batch_size = self.mu.shape[0]

    @lazy_property
    def greedy_actions(self):
        a = self.mu
        if self.min_action is not None:
            a = torch.max(self.min_action.unsqueeze(0).expand_as(a), a)
        if self.max_action is not None:
            a = torch.min(self.max_action.unsqueeze(0).expand_as(a), a)
        return a

    @lazy_property
    def max(self):
        if self.min_action is None and self.max_action is None:
            return self.v.reshape(
                self.batch_size,
            )
        else:
            return self.evaluate_actions(self.greedy_actions)

    def evaluate_actions(self, actions):
        u_minus_mu = actions - self.mu
        a = (
            -0.5
            * torch.matmul(
                torch.matmul(u_minus_mu[:, None, :], self.mat), u_minus_mu[:, :, None]
            )[:, 0, 0]
        )
        return a + self.v.reshape(
            self.batch_size,
        )

    def compute_advantage(self, actions):
        return self.evaluate_actions(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return self.evaluate_actions(actions) - self.evaluate_actions(argmax_actions)

    def __repr__(self):
        return "QuadraticActionValue greedy_actions:{} v:{}".format(
            self.greedy_actions.detach().cpu().numpy(), self.v.detach().cpu().numpy()
        )

    @property
    def params(self):
        return (self.mu, self.mat, self.v)

    def __getitem__(self, i):
        return QuadraticActionValue(
            self.mu[i],
            self.mat[i],
            self.v[i],
            min_action=self.min_action,
            max_action=self.max_action,
        )

class SingleActionValue(ActionValue):
    """ActionValue that can evaluate only a single action."""

    def __init__(self, evaluator, maximizer=None):
        self.evaluator = evaluator
        self.maximizer = maximizer

    @lazy_property
    def greedy_actions(self):
        return self.maximizer()

    @lazy_property
    def max(self):
        return self.evaluator(self.greedy_actions)

    def evaluate_actions(self, actions):
        return self.evaluator(actions)

    def compute_advantage(self, actions):
        return self.evaluator(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return self.evaluate_actions(actions) - self.evaluate_actions(argmax_actions)

    def __repr__(self):
        return "SingleActionValue"

    @property
    def params(self):
        warnings.warn(
            "SingleActionValue has no learnable parameters until it"
            " is evaluated on some action. If you want to draw a computation"
            " graph that outputs SingleActionValue, use the variable returned"
            " by its method such as evaluate_actions instead."
        )
        return ()

    def __getitem__(self, i):
        raise NotImplementedError

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

