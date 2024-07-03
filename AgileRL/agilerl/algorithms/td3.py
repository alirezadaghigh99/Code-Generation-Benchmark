    def learn(self, experiences, noise_clip=0.5, policy_noise=0.2):
        """Updates agent network parameters to learn from experiences.

        :param experience: List of batched states, actions, rewards, next_states, dones in that order.
        :type experience: list[torch.Tensor[float]]
        :param noise_clip: Maximum noise limit to apply to actions, defaults to 0.5
        :type noise_clip: float, optional
        :param policy_noise: Standard deviation of noise applied to policy, defaults to 0.2
        :type policy_noise: float, optional
        """
        states, actions, rewards, next_states, dones = experiences
        if self.accelerator is not None:
            states = states.to(self.accelerator.device)
            actions = actions.to(self.accelerator.device)
            rewards = rewards.to(self.accelerator.device)
            next_states = next_states.to(self.accelerator.device)
            dones = dones.to(self.accelerator.device)

        if self.one_hot:
            states = (
                nn.functional.one_hot(states.long(), num_classes=self.state_dim[0])
                .float()
                .squeeze()
            )
            next_states = (
                nn.functional.one_hot(next_states.long(), num_classes=self.state_dim[0])
                .float()
                .squeeze()
            )

        if self.arch == "mlp":
            input_combined = torch.cat([states, actions], 1)
            q_value_1 = self.critic_1(input_combined)
            q_value_2 = self.critic_2(input_combined)
        elif self.arch == "cnn":
            q_value_1 = self.critic_1(states, actions)
            q_value_2 = self.critic_2(states, actions)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # Scale actions
            next_actions = torch.where(
                next_actions > 0,
                next_actions * self.max_action,
                next_actions * -self.min_action,
            )
            noise = actions.data.normal_(0, policy_noise)
            if self.accelerator is not None:
                noise = noise.to(self.accelerator.device)
            else:
                noise = noise.to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_actions = next_actions + noise

            # Compute the target, y_j, making use of twin critic networks
            if self.arch == "mlp":
                next_input_combined = torch.cat([next_states, next_actions], 1)
                q_value_next_state_1 = self.critic_target_1(next_input_combined)
                q_value_next_state_2 = self.critic_target_2(next_input_combined)
            elif self.arch == "cnn":
                q_value_next_state_1 = self.critic_target_1(next_states, next_actions)
                q_value_next_state_2 = self.critic_target_2(next_states, next_actions)
            q_value_next_state = torch.min(q_value_next_state_1, q_value_next_state_2)

        y_j = rewards + ((1 - dones) * self.gamma * q_value_next_state)

        # Loss equation needs to be updated to account for two q_values from two critics
        critic_loss = self.criterion(q_value_1, y_j) + self.criterion(q_value_2, y_j)

        # critic loss backprop
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(critic_loss)
        else:
            critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # update actor and targets every policy_freq learn steps
        self.learn_counter += 1
        if self.learn_counter % self.policy_freq == 0:
            policy_actions = self.actor.forward(states)
            policy_actions = torch.where(
                policy_actions > 0,
                policy_actions * self.max_action,
                policy_actions * -self.min_action,
            )
            if self.arch == "mlp":
                input_combined = torch.cat([states, policy_actions], 1)
                actor_loss = -self.critic_1(input_combined).mean()
            elif self.arch == "cnn":
                actor_loss = -self.critic_1(states, policy_actions).mean()

            # actor loss backprop
            self.actor_optimizer.zero_grad()
            if self.accelerator is not None:
                self.accelerator.backward(actor_loss)
            else:
                actor_loss.backward()
            self.actor_optimizer.step()

            # Add in a soft update for both critic_targets
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic_1, self.critic_target_1)
            self.soft_update(self.critic_2, self.critic_target_2)

            return actor_loss.item(), critic_loss.item()
        else:
            return None, critic_loss.item()