class Adam(Optimizer):
    """The Adam optimization algorithm."""

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08,
                 weight_decay: float = 0):
        """Construct an Adam optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        beta1: float
            a parameter of the Adam algorithm
        beta2: float
            a parameter of the Adam algorithm
        epsilon: float
            a parameter of the Adam algorithm
        weight_decay: float
            L2 penalty - a parameter of the Adam algorithm
        """
        super(Adam, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def _create_tf_optimizer(self, global_step):
        import tensorflow as tf
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate,
                                               beta_1=self.beta1,
                                               beta_2=self.beta2,
                                               epsilon=self.epsilon)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.Adam(params,
                                lr=lr,
                                betas=(self.beta1, self.beta2),
                                eps=self.epsilon,
                                weight_decay=self.weight_decay)

    def _create_jax_optimizer(self):
        import optax
        process = []
        if isinstance(self.learning_rate, LearningRateSchedule):
            scheduler = self.learning_rate._create_jax_schedule()
            process.append(optax.scale_by_schedule(scheduler))
            last_process = optax.scale(-1.0)
        else:
            lr = self.learning_rate
            last_process = optax.scale(-1.0 * lr)

        process.append(
            optax.scale_by_adam(b1=self.beta1, b2=self.beta2, eps=self.epsilon))
        process.append(last_process)
        return optax.chain(*process)class AdamW(Optimizer):
    """The AdamW optimization algorithm.
    AdamW is a variant of Adam, with improved weight decay.
    In Adam, weight decay is implemented as: weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
    In AdamW, weight decay is implemented as: weight_decay (float, optional) – weight decay coefficient (default: 1e-2)
    """

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 weight_decay: Union[float, LearningRateSchedule] = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08,
                 amsgrad: bool = False):
        """Construct an AdamW optimizer.
        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        weight_decay: float or LearningRateSchedule
            weight decay coefficient for AdamW
        beta1: float
            a parameter of the Adam algorithm
        beta2: float
            a parameter of the Adam algorithm
        epsilon: float
            a parameter of the Adam algorithm
        amsgrad: bool
            If True, will use the AMSGrad variant of AdamW (from "On the Convergence of Adam and Beyond"), else will use the original algorithm.
        """
        super(AdamW, self).__init__(learning_rate)
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def _create_tf_optimizer(self, global_step):
        import tensorflow_addons as tfa
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tfa.optimizers.AdamW(weight_decay=self.weight_decay,
                                    learning_rate=learning_rate,
                                    beta_1=self.beta1,
                                    beta_2=self.beta2,
                                    epsilon=self.epsilon,
                                    amsgrad=self.amsgrad)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.AdamW(params, lr, (self.beta1, self.beta2),
                                 self.epsilon, self.weight_decay, self.amsgrad)

    def _create_jax_optimizer(self):
        import optax
        process = []
        if isinstance(self.learning_rate, LearningRateSchedule):
            scheduler = self.learning_rate._create_jax_schedule()
            process.append(optax.scale_by_schedule(scheduler))
            last_process = optax.scale(-1.0)
        else:
            lr = self.learning_rate
            last_process = optax.scale(-1.0 * lr)

        process.append(
            optax.scale_by_adam(b1=self.beta1,
                                b2=self.beta2,
                                eps=self.epsilon,
                                eps_root=0.0))
        process.append(optax.add_decayed_weights(self.weight_decay, None))
        process.append(last_process)
        return optax.chain(*process)class AdaGrad(Optimizer):
    """The AdaGrad optimization algorithm.

    Adagrad is an optimizer with parameter-specific learning rates, which are
    adapted relative to how frequently a parameter gets updated during training.
    The more updates a parameter receives, the smaller the updates. See [1]_ for
    a full reference for the algorithm.

    References
    ----------
    .. [1] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient
        methods for online learning and stochastic optimization." Journal of machine
        learning research 12.7 (2011).
    """

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 initial_accumulator_value: float = 0.1,
                 epsilon: float = 1e-07):
        """Construct an AdaGrad optimizer.
        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        initial_accumulator_value: float
            a parameter of the AdaGrad algorithm
        epsilon: float
            a parameter of the AdaGrad algorithm

        """
        super(AdaGrad, self).__init__(learning_rate)
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

    def _create_tf_optimizer(self, global_step):
        import tensorflow as tf
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tf.keras.optimizers.legacy.Adagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=self.initial_accumulator_value,
            epsilon=self.epsilon)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.Adagrad(
            params,
            lr,
            initial_accumulator_value=self.initial_accumulator_value,
            eps=self.epsilon)

    def _create_jax_optimizer(self):
        import optax
        process = []
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
            last_process = optax.scale(-1.0)
        else:
            lr = self.learning_rate
            last_process = optax.scale(-1.0 * lr)

        process.append(
            optax.scale_by_rss(
                initial_accumulator_value=self.initial_accumulator_value,
                eps=self.epsilon))
        process.append(last_process)
        return optax.chain(*process)class RMSProp(Optimizer):
    """RMSProp Optimization algorithm."""

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 momentum: float = 0.0,
                 decay: float = 0.9,
                 epsilon: float = 1e-10):
        """Construct an RMSProp Optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning_rate used for optimization
        momentum: float, default 0.0
            a parameter of the RMSProp algorithm
        decay: float, default 0.9
            a parameter of the RMSProp algorithm
        epsilon: float, default 1e-10
            a parameter of the RMSProp algorithm
        """
        super(RMSProp, self).__init__(learning_rate)
        self.momentum = momentum
        self.decay = decay
        self.epsilon = epsilon

    def _create_tf_optimizer(self, global_step):
        import tensorflow as tf
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate,
                                                  momentum=self.momentum,
                                                  rho=self.decay,
                                                  epsilon=self.epsilon)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.RMSprop(params,
                                   lr,
                                   alpha=self.decay,
                                   eps=self.epsilon,
                                   momentum=self.momentum)

    def _create_jax_optimizer(self):
        import optax
        process = []
        if isinstance(self.learning_rate, LearningRateSchedule):
            scheduler = self.learning_rate._create_jax_schedule()
            process.append(optax.scale_by_schedule(scheduler))
            last_process = optax.scale(-1.0)
        else:
            lr = self.learning_rate
            last_process = optax.scale(-1.0 * lr)

        process.append(
            optax.scale_by_rms(decay=self.decay,
                               eps=self.epsilon,
                               initial_scale=0.0))
        if self.momentum is not None or self.momentum != 0.0:
            process.append(optax.trace(decay=self.momentum, nesterov=False))
        process.append(last_process)
        return optax.chain(*process)class GradientDescent(Optimizer):
    """The gradient descent optimization algorithm."""

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001):
        """Construct a gradient descent optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        """
        super(GradientDescent, self).__init__(learning_rate)

    def _create_tf_optimizer(self, global_step):
        import tensorflow as tf
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.SGD(params, lr)

    def _create_jax_optimizer(self):
        import optax
        process = []
        if isinstance(self.learning_rate, LearningRateSchedule):
            scheduler = self.learning_rate._create_jax_schedule()
            process.append(optax.scale_by_schedule(scheduler))
            last_process = optax.scale(-1.0)
        else:
            lr = self.learning_rate
            last_process = optax.scale(-1.0 * lr)
        process.append(last_process)
        return optax.chain(*process)class SparseAdam(Optimizer):
    """The Sparse Adam optimization algorithm, also known as Lazy Adam.
    Sparse Adam is suitable for sparse tensors. It handles sparse updates more efficiently.
    It only updates moving-average accumulators for sparse variable indices that appear in the current batch, rather than updating the accumulators for all indices.
    """

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08):
        """Construct an Adam optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        beta1: float
            a parameter of the SparseAdam algorithm
        beta2: float
            a parameter of the SparseAdam algorithm
        epsilon: float
            a parameter of the SparseAdam algorithm
        """
        super(SparseAdam, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_tf_optimizer(self, global_step):
        import tensorflow_addons as tfa
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tfa.optimizers.LazyAdam(learning_rate=learning_rate,
                                       beta_1=self.beta1,
                                       beta_2=self.beta2,
                                       epsilon=self.epsilon)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.SparseAdam(params, lr, (self.beta1, self.beta2),
                                      self.epsilon)