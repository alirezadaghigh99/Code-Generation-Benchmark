class Agent:
    """Avalanche Continual Learning Agent.

    The agent stores the state needed by continual learning training methods,
    such as optimizers, models, regularization losses.
    You can add any objects as attributes dynamically:

    .. code-block::

        agent = Agent()
        agent.replay = ReservoirSamplingBuffer(max_size=200)
        agent.loss = MaskedCrossEntropy()
        agent.reg_loss = LearningWithoutForgetting(alpha=1, temperature=2)
        agent.model = my_model
        agent.opt = SGD(agent.model.parameters(), lr=0.001)
        agent.scheduler = ExponentialLR(agent.opt, gamma=0.999)

    Many CL objects will need to perform some operation before or
    after training on each experience. This is supported via the `Adaptable`
    Protocol, which requires the `pre_adapt` and `post_adapt` methods.
    To call the pre/post adaptation you can implement your training loop
    like in the following example:

    .. code-block::

        def train(agent, exp):
            agent.pre_adapt(exp)
            # do training here
            agent.post_adapt(exp)

    Objects that implement the `Adaptable` Protocol will be called by the Agent.

    You can also add additional functionality to the adaptation phases with
    hooks. For example:

    .. code-block::
        agent.add_pre_hooks(lambda a, e: update_optimizer(a.opt, new_params={}, optimized_params=dict(a.model.named_parameters())))
        # we update the lr scheduler after each experience (not every epoch!)
        agent.add_post_hooks(lambda a, e: a.scheduler.step())


    """

    def __init__(self, verbose=False):
        """Init.

        :param verbose: If True, print every time an adaptable object or hook
            is called during the adaptation. Useful for debugging.
        """
        self._updatable_objects = []
        self.verbose = verbose
        self._pre_hooks = []
        self._post_hooks = []

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(value, "pre_adapt") or hasattr(value, "post_adapt"):
            self._updatable_objects.append(value)
            if self.verbose:
                print("Added updatable object ", value)

    def pre_adapt(self, exp):
        """Pre-adaptation.

        Remember to call this before training on a new experience.

        :param exp: current experience
        """
        for uo in self._updatable_objects:
            if hasattr(uo, "pre_adapt"):
                uo.pre_adapt(self, exp)
                if self.verbose:
                    print("pre_adapt ", uo)
        for foo in self._pre_hooks:
            if self.verbose:
                print("pre_adapt hook ", foo)
            foo(self, exp)

    def post_adapt(self, exp):
        """Post-adaptation.

        Remember to call this after training on a new experience.

        :param exp: current experience
        """
        for uo in self._updatable_objects:
            if hasattr(uo, "post_adapt"):
                uo.post_adapt(self, exp)
                if self.verbose:
                    print("post_adapt ", uo)
        for foo in self._post_hooks:
            if self.verbose:
                print("post_adapt hook ", foo)
            foo(self, exp)

    def add_pre_hooks(self, foo):
        """Add a pre-adaptation hooks

        Hooks take two arguments: `<agent, experience>`.

        :param foo: the hook function
        """
        self._pre_hooks.append(foo)

    def add_post_hooks(self, foo):
        """Add a post-adaptation hooks

        Hooks take two arguments: `<agent, experience>`.

        :param foo: the hook function
        """
        self._post_hooks.append(foo)

