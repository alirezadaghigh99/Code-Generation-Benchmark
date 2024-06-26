    def mutation(self, population, pre_training_mut=False):
        """Returns mutated population.

        :param population: Population of agents
        :type population: list[object]
        :param pre_training_mut: Boolean flag indicating if the mutation is before the training loop
        :type pre_training_mut: bool, optional
        """
        # Create lists of possible mutation functions and their respective
        # relative probabilities
        mutation_options = []
        mutation_proba = []
        if self.no_mut:
            mutation_options.append(self.no_mutation)
            if pre_training_mut:
                mutation_proba.append(float(0))
            else:
                mutation_proba.append(float(self.no_mut))
        if self.architecture_mut:
            mutation_options.append(self.architecture_mutate)
            mutation_proba.append(float(self.architecture_mut))
        if self.parameters_mut:
            mutation_options.append(self.parameter_mutation)
            mutation_proba.append(float(self.parameters_mut))
        if self.activation_mut:
            mutation_options.append(self.activation_mutation)
            mutation_proba.append(float(self.activation_mut))
        if self.rl_hp_mut:
            mutation_options.append(self.rl_hyperparam_mutation)
            mutation_proba.append(float(self.rl_hp_mut))

        if len(mutation_options) == 0:  # Return if no mutation options
            return population

        mutation_proba = np.array(mutation_proba) / np.sum(
            mutation_proba
        )  # Normalize probs

        # Randomly choose mutation for each agent in population from options with
        # relative probabilities
        mutation_choice = self.rng.choice(
            mutation_options, len(population), p=mutation_proba
        )

        # If not mutating elite member of population (first in list from tournament selection),
        # set this as the first mutation choice
        if not self.mutate_elite:
            mutation_choice[0] = self.no_mutation

        mutated_population = []
        for mutation, individual in zip(mutation_choice, population):
            # Call mutation function for each individual
            individual = mutation(individual)

            if self.multi_agent:
                offspring_actors = getattr(individual, self.algo["actor"]["eval"])

                # Reinitialise target network with frozen weights due to potential
                # mutation in architecture of value network
                ind_targets = [
                    type(offspring_actor)(**offspring_actor.init_dict)
                    for offspring_actor in offspring_actors
                ]

                for ind_target, offspring_actor in zip(ind_targets, offspring_actors):
                    ind_target.load_state_dict(offspring_actor.state_dict())

                if self.accelerator is None:
                    ind_targets = [
                        ind_target.to(self.device) for ind_target in ind_targets
                    ]
                setattr(individual, self.algo["actor"]["target"], ind_targets)

                # If algorithm has critics, reinitialize their respective target networks
                # too
                for critics_list in self.algo["critics"]:
                    offspring_critics = getattr(individual, critics_list["eval"])
                    ind_targets = [
                        type(offspring_critic)(**offspring_critic.init_dict)
                        for offspring_critic in offspring_critics
                    ]

                    for ind_target, offspring_critic in zip(
                        ind_targets, offspring_critics
                    ):
                        ind_target.load_state_dict(offspring_critic.state_dict())

                    if self.accelerator is None:
                        ind_targets = [
                            ind_target.to(self.device) for ind_target in ind_targets
                        ]
                    setattr(individual, critics_list["target"], ind_targets)
            else:
                if "target" in self.algo["actor"].keys():
                    offspring_actor = getattr(individual, self.algo["actor"]["eval"])

                    # Reinitialise target network with frozen weights due to potential
                    # mutation in architecture of value network
                    ind_target = type(offspring_actor)(**offspring_actor.init_dict)
                    ind_target.load_state_dict(offspring_actor.state_dict())
                    if self.accelerator is not None:
                        setattr(individual, self.algo["actor"]["target"], ind_target)
                    else:
                        setattr(
                            individual,
                            self.algo["actor"]["target"],
                            ind_target.to(self.device),
                        )

                    # If algorithm has critics, reinitialize their respective target networks
                    # too
                    for critic in self.algo["critics"]:
                        offspring_critic = getattr(individual, critic["eval"])
                        ind_target = type(offspring_critic)(
                            **offspring_critic.init_dict
                        )
                        ind_target.load_state_dict(offspring_critic.state_dict())
                        if self.accelerator is not None:
                            setattr(individual, critic["target"], ind_target)
                        else:
                            setattr(
                                individual, critic["target"], ind_target.to(self.device)
                            )

            mutated_population.append(individual)

        return mutated_population