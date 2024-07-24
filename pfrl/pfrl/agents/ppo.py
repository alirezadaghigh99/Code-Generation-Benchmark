def _yield_minibatches(dataset, minibatch_size, num_epochs):
    assert dataset
    buf = []
    n = 0
    while n < len(dataset) * num_epochs:
        while len(buf) < minibatch_size:
            buf = random.sample(dataset, k=len(dataset)) + buf
        assert len(buf) >= minibatch_size
        yield buf[-minibatch_size:]
        n += minibatch_size
        buf = buf[:-minibatch_size]

def _limit_sequence_length(sequences, max_len):
    assert max_len > 0
    new_sequences = []
    for sequence in sequences:
        while len(sequence) > max_len:
            new_sequences.append(sequence[:max_len])
            sequence = sequence[max_len:]
        assert 0 < len(sequence) <= max_len
        new_sequences.append(sequence)
    return new_sequences

class PPO(agent.AttributeSavingMixin, agent.BatchAgent):
    """Proximal Policy Optimization

    See https://arxiv.org/abs/1707.06347

    Args:
        model (torch.nn.Module): Model to train (including recurrent models)
            state s  |->  (pi(s, _), v(s))
        optimizer (torch.optim.Optimizer): Optimizer used to train the model
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        value_func_coef (float): Weight coefficient for loss of
            value function (0, inf)
        entropy_coef (float): Weight coefficient for entropy bonus [0, inf)
        update_interval (int): Model update interval in step
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        clip_eps (float): Epsilon for pessimistic clipping of likelihood ratio
            to update policy
        clip_eps_vf (float): Epsilon for pessimistic clipping of value
            to update value function. If it is ``None``, value function is not
            clipped on updates.
        standardize_advantages (bool): Use standardized advantages on updates
        recurrent (bool): If set to True, `model` is assumed to implement
            `pfrl.nn.Recurrent` and update in a recurrent
            manner.
        max_recurrent_sequence_len (int): Maximum length of consecutive
            sequences of transitions in a minibatch for updating the model.
            This value is used only when `recurrent` is True. A smaller value
            will encourage a minibatch to contain more and shorter sequences.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        value_stats_window (int): Window size used to compute statistics
            of value predictions.
        entropy_stats_window (int): Window size used to compute statistics
            of entropy of action distributions.
        value_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the value function.
        policy_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the policy.

    Statistics:
        average_value: Average of value predictions on non-terminal states.
            It's updated on (batch_)act_and_train.
        average_entropy: Average of entropy of action distributions on
            non-terminal states. It's updated on (batch_)act_and_train.
        average_value_loss: Average of losses regarding the value function.
            It's updated after the model is updated.
        average_policy_loss: Average of losses regarding the policy.
            It's updated after the model is updated.
        n_updates: Number of model updates so far.
        explained_variance: Explained variance computed from the last batch.
    """

    saved_attributes = ("model", "optimizer", "obs_normalizer")

    def __init__(
        self,
        model,
        optimizer,
        obs_normalizer=None,
        gpu=None,
        gamma=0.99,
        lambd=0.95,
        phi=lambda x: x,
        value_func_coef=1.0,
        entropy_coef=0.01,
        update_interval=2048,
        minibatch_size=64,
        epochs=10,
        clip_eps=0.2,
        clip_eps_vf=None,
        standardize_advantages=True,
        batch_states=batch_states,
        recurrent=False,
        max_recurrent_sequence_len=None,
        act_deterministically=False,
        max_grad_norm=None,
        value_stats_window=1000,
        entropy_stats_window=1000,
        value_loss_stats_window=100,
        policy_loss_stats_window=100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.obs_normalizer = obs_normalizer

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
            if self.obs_normalizer is not None:
                self.obs_normalizer.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.value_func_coef = value_func_coef
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.clip_eps_vf = clip_eps_vf
        self.standardize_advantages = standardize_advantages
        self.batch_states = batch_states
        self.recurrent = recurrent
        self.max_recurrent_sequence_len = max_recurrent_sequence_len
        self.act_deterministically = act_deterministically
        self.max_grad_norm = max_grad_norm

        # Contains episodes used for next update iteration
        self.memory = []

        # Contains transitions of the last episode not moved to self.memory yet
        self.last_episode = []
        self.last_state = None
        self.last_action = None

        # Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.value_loss_record = collections.deque(maxlen=value_loss_stats_window)
        self.policy_loss_record = collections.deque(maxlen=policy_loss_stats_window)
        self.explained_variance = np.nan
        self.n_updates = 0

    def _initialize_batch_variables(self, num_envs):
        self.batch_last_episode = [[] for _ in range(num_envs)]
        self.batch_last_state = [None] * num_envs
        self.batch_last_action = [None] * num_envs

    def _update_if_dataset_is_ready(self):
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (
                0
                if self.batch_last_episode is None
                else sum(len(episode) for episode in self.batch_last_episode)
            )
        )
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            if self.recurrent:
                dataset = _make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                    device=self.device,
                )
                self._update_recurrent(dataset)
            else:
                dataset = _make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    device=self.device,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            self.explained_variance = _compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory))
            )
            self.memory = []

    def _flush_last_episode(self):
        if self.last_episode:
            self.memory.append(self.last_episode)
            self.last_episode = []
        if self.batch_last_episode:
            for i, episode in enumerate(self.batch_last_episode):
                if episode:
                    self.memory.append(episode)
                    self.batch_last_episode[i] = []

    def _update_obs_normalizer(self, dataset):
        assert self.obs_normalizer
        states = self.batch_states([b["state"] for b in dataset], self.device, self.phi)
        self.obs_normalizer.experience(states)

    def _update(self, dataset):
        """Update both the policy and the value function."""

        device = self.device

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

        assert "state" in dataset[0]
        assert "v_teacher" in dataset[0]

        if self.standardize_advantages:
            all_advs = torch.tensor([b["adv"] for b in dataset], device=device)
            std_advs, mean_advs = torch.std_mean(all_advs, unbiased=False)

        for batch in _yield_minibatches(
            dataset, minibatch_size=self.minibatch_size, num_epochs=self.epochs
        ):
            states = self.batch_states(
                [b["state"] for b in batch], self.device, self.phi
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            actions = torch.tensor([b["action"] for b in batch], device=device)
            distribs, vs_pred = self.model(states)

            advs = torch.tensor(
                [b["adv"] for b in batch], dtype=torch.float32, device=device
            )
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            log_probs_old = torch.tensor(
                [b["log_prob"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            vs_pred_old = torch.tensor(
                [b["v_pred"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            vs_teacher = torch.tensor(
                [b["v_teacher"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            # Same shape as vs_pred: (batch_size, 1)
            vs_pred_old = vs_pred_old[..., None]
            vs_teacher = vs_teacher[..., None]

            self.model.zero_grad()
            loss = self._lossfun(
                distribs.entropy(),
                vs_pred,
                distribs.log_prob(actions),
                vs_pred_old=vs_pred_old,
                log_probs_old=log_probs_old,
                advs=advs,
                vs_teacher=vs_teacher,
            )
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            self.n_updates += 1

    def _update_once_recurrent(self, episodes, mean_advs, std_advs):
        assert std_advs is None or std_advs > 0

        device = self.device

        # Sort desc by lengths so that pack_sequence does not change the order
        episodes = sorted(episodes, key=len, reverse=True)

        flat_transitions = flatten_sequences_time_first(episodes)

        # Prepare data for a recurrent model
        seqs_states = []
        for ep in episodes:
            states = self.batch_states(
                [transition["state"] for transition in ep],
                self.device,
                self.phi,
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            seqs_states.append(states)

        flat_actions = torch.tensor(
            [transition["action"] for transition in flat_transitions],
            device=device,
        )
        flat_advs = torch.tensor(
            [transition["adv"] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )
        if self.standardize_advantages:
            flat_advs = (flat_advs - mean_advs) / (std_advs + 1e-8)
        flat_log_probs_old = torch.tensor(
            [transition["log_prob"] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )
        flat_vs_pred_old = torch.tensor(
            [[transition["v_pred"]] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )
        flat_vs_teacher = torch.tensor(
            [[transition["v_teacher"]] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            rs = concatenate_recurrent_states(
                [ep[0]["recurrent_state"] for ep in episodes]
            )

        (flat_distribs, flat_vs_pred), _ = pack_and_forward(self.model, seqs_states, rs)
        flat_log_probs = flat_distribs.log_prob(flat_actions)
        flat_entropy = flat_distribs.entropy()

        self.model.zero_grad()
        loss = self._lossfun(
            entropy=flat_entropy,
            vs_pred=flat_vs_pred,
            log_probs=flat_log_probs,
            vs_pred_old=flat_vs_pred_old,
            log_probs_old=flat_log_probs_old,
            advs=flat_advs,
            vs_teacher=flat_vs_teacher,
        )
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.n_updates += 1

    def _update_recurrent(self, dataset):
        """Update both the policy and the value function."""

        device = self.device

        flat_dataset = list(itertools.chain.from_iterable(dataset))
        if self.obs_normalizer:
            self._update_obs_normalizer(flat_dataset)

        assert "state" in flat_dataset[0]
        assert "v_teacher" in flat_dataset[0]

        if self.standardize_advantages:
            all_advs = torch.tensor([b["adv"] for b in flat_dataset], device=device)
            std_advs, mean_advs = torch.std_mean(all_advs, unbiased=False)
        else:
            mean_advs = None
            std_advs = None

        for _ in range(self.epochs):
            random.shuffle(dataset)
            for minibatch in _yield_subset_of_sequences_with_fixed_number_of_items(
                dataset, self.minibatch_size
            ):
                self._update_once_recurrent(minibatch, mean_advs, std_advs)

    def _lossfun(
        self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher
    ):
        prob_ratio = torch.exp(log_probs - log_probs_old)

        loss_policy = -torch.mean(
            torch.min(
                prob_ratio * advs,
                torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs,
            ),
        )

        if self.clip_eps_vf is None:
            loss_value_func = F.mse_loss(vs_pred, vs_teacher)
        else:
            clipped_vs_pred = _elementwise_clip(
                vs_pred,
                vs_pred_old - self.clip_eps_vf,
                vs_pred_old + self.clip_eps_vf,
            )
            loss_value_func = torch.mean(
                torch.max(
                    F.mse_loss(vs_pred, vs_teacher, reduction="none"),
                    F.mse_loss(clipped_vs_pred, vs_teacher, reduction="none"),
                )
            )
        loss_entropy = -torch.mean(entropy)

        self.value_loss_record.append(float(loss_value_func))
        self.policy_loss_record.append(float(loss_policy))

        loss = (
            loss_policy
            + self.value_func_coef * loss_value_func
            + self.entropy_coef * loss_entropy
        )

        return loss

    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset)
        else:
            self._batch_observe_eval(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_act_eval(self, batch_obs):
        assert not self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                (action_distrib, _), self.test_recurrent_states = one_step_forward(
                    self.model, b_state, self.test_recurrent_states
                )
            else:
                action_distrib, _ = self.model(b_state)
            if self.act_deterministically:
                action = mode_of_distribution(action_distrib).cpu().numpy()
            else:
                action = action_distrib.sample().cpu().numpy()

        return action

    def _batch_act_train(self, batch_obs):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs

        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (
                    (action_distrib, batch_value),
                    self.train_recurrent_states,
                ) = one_step_forward(
                    self.model, b_state, self.train_prev_recurrent_states
                )
            else:
                action_distrib, batch_value = self.model(b_state)
            batch_action = action_distrib.sample().cpu().numpy()
            self.entropy_record.extend(action_distrib.entropy().cpu().numpy())
            self.value_record.extend(batch_value.cpu().numpy())

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def _batch_observe_eval(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert not self.training
        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.test_recurrent_states = mask_recurrent_state_at(
                    self.test_recurrent_states, indices_that_ended
                )

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert self.training

        for i, (state, action, reward, next_state, done, reset) in enumerate(
            zip(
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset,
            )
        ):
            if state is not None:
                assert action is not None
                transition = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "nonterminal": 0.0 if done else 1.0,
                }
                if self.recurrent:
                    transition["recurrent_state"] = get_recurrent_state_at(
                        self.train_prev_recurrent_states, i, detach=True
                    )
                    transition["next_recurrent_state"] = get_recurrent_state_at(
                        self.train_recurrent_states, i, detach=True
                    )
                self.batch_last_episode[i].append(transition)
            if done or reset:
                assert self.batch_last_episode[i]
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
            self.batch_last_state[i] = None
            self.batch_last_action[i] = None

        self.train_prev_recurrent_states = None

        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.train_recurrent_states = mask_recurrent_state_at(
                    self.train_recurrent_states, indices_that_ended
                )

        self._update_if_dataset_is_ready()

    def get_statistics(self):
        return [
            ("average_value", _mean_or_nan(self.value_record)),
            ("average_entropy", _mean_or_nan(self.entropy_record)),
            ("average_value_loss", _mean_or_nan(self.value_loss_record)),
            ("average_policy_loss", _mean_or_nan(self.policy_loss_record)),
            ("n_updates", self.n_updates),
            ("explained_variance", self.explained_variance),
        ]

