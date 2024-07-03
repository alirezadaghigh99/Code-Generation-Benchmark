    def reinit_opt(self, individual):
        if self.multi_agent:
            # Reinitialise optimizer
            actor_opts = getattr(individual, self.algo["actor"]["optimizer"])

            net_params = [
                actor.parameters()
                for actor in getattr(individual, self.algo["actor"]["eval"])
            ]

            offspring_actor_opts = [
                type(actor_opt)(net_param, lr=individual.lr_actor)
                for actor_opt, net_param in zip(actor_opts, net_params)
            ]

            setattr(
                individual,
                self.algo["actor"]["optimizer"],
                offspring_actor_opts,
            )

            for critic_list in self.algo["critics"]:
                critic_opts = getattr(individual, critic_list["optimizer"])

                net_params = [
                    critic.parameters()
                    for critic in getattr(individual, critic_list["eval"])
                ]

                offspring_critic_opts = [
                    type(critic_opt)(net_param, lr=individual.lr_critic)
                    for critic_opt, net_param in zip(critic_opts, net_params)
                ]

                setattr(
                    individual,
                    critic_list["optimizer"],
                    offspring_critic_opts,
                )
        else:
            if individual.algo in ["PPO"]:
                # Reinitialise optimizer
                opt = getattr(individual, self.algo["actor"]["optimizer"])
                actor_net_params = getattr(
                    individual, self.algo["actor"]["eval"]
                ).parameters()
                critic_net_params = getattr(
                    individual, self.algo["critics"][0]["eval"]
                ).parameters()
                opt_args = [
                    {"params": actor_net_params, "lr": individual.lr},
                    {"params": critic_net_params, "lr": individual.lr},
                ]
                setattr(
                    individual,
                    self.algo["actor"]["optimizer"],
                    type(opt)(opt_args),
                )

            else:
                # Reinitialise optimizer
                actor_opt = getattr(individual, self.algo["actor"]["optimizer"])
                net_params = getattr(
                    individual, self.algo["actor"]["eval"]
                ).parameters()
                if individual.algo in ["DDPG", "TD3"]:
                    setattr(
                        individual,
                        self.algo["actor"]["optimizer"],
                        type(actor_opt)(net_params, lr=individual.lr_actor),
                    )
                else:
                    setattr(
                        individual,
                        self.algo["actor"]["optimizer"],
                        type(actor_opt)(net_params, lr=individual.lr),
                    )

                # If algorithm has critics, reinitialise their optimizers too
                for critic in self.algo["critics"]:
                    critic_opt = getattr(individual, critic["optimizer"])
                    net_params = getattr(individual, critic["eval"]).parameters()
                    setattr(
                        individual,
                        critic["optimizer"],
                        type(critic_opt)(net_params, lr=individual.lr_critic),
                    )