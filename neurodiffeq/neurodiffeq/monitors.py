class Monitor1D(BaseMonitor):
    """A monitor for checking the status of the neural network during training.

    :param t_min:
        The lower bound of time domain that we want to monitor.
    :type t_min: float
    :param t_max:
        The upper bound of time domain that we want to monitor.
    :type t_max: float
    :param check_every:
        The frequency of checking the neural network represented by the number of epochs between two checks.
        Defaults to 100.
    :type check_every: int, optional
    """

    def __init__(self, t_min, t_max, check_every=None):
        """Initializer method
        """
        super(Monitor1D, self).__init__(check_every=check_every)
        self.fig = plt.figure(figsize=(30, 8))
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        # input for plotting
        self.ts_plt = np.linspace(t_min, t_max, 100)
        # input for neural network
        self.ts_ann = torch.linspace(t_min, t_max, 100, requires_grad=True).reshape((-1, 1))

    def check(self, nets, conditions, history):
        r"""Draw 2 plots: One shows the shape of the current solution.
        The other shows the history training loss and validation loss.

        :param nets:
            The neural networks that approximates the ODE (system).
        :type nets: list[`torch.nn.Module`]
        :param conditions:
            The initial/boundary conditions of the ODE (system).
        :type conditions: list[`neurodiffeq.ode.BaseCondition`]
        :param history:
            The history of training loss and validation loss.
            The 'train_loss' entry is a list of training loss and 'valid_loss' entry is a list of validation loss.
        :type history: dict['train': list[float], 'valid': list[float]]

        .. note::
            `check` is meant to be called by the function `solve` and `solve_system`.
        """
        us = [
            cond.enforce(net, self.ts_ann).detach().cpu().numpy()
            for cond, net in zip(conditions, nets)
        ]

        self.ax1.clear()
        for i, u in enumerate(us):
            self.ax1.plot(self.ts_plt, u, label=f'variable {i}')
        self.ax1.legend()
        self.ax1.set_title('solutions')

        self.ax2.clear()
        self.ax2.plot(history['train_loss'], label='training loss')
        self.ax2.plot(history['valid_loss'], label='validation loss')
        self.ax2.set_title('loss during training')
        self.ax2.set_ylabel('loss')
        self.ax2.set_xlabel('epochs')
        self.ax2.set_yscale('log')
        self.ax2.legend()

        self.ax3.clear()
        for metric_name, metric_values in history.items():
            if metric_name == 'train_loss' or metric_name == 'valid_loss':
                continue
            self.ax3.plot(metric_values, label=metric_name)
        self.ax3.set_title('metrics during training')
        self.ax3.set_ylabel('metrics')
        self.ax3.set_xlabel('epochs')
        self.ax3.set_yscale('log')
        # if there's not custom metrics, then there won't be any labels in this axis
        if len(history) > 2:
            self.ax3.legend()

        self.fig.canvas.draw()
        if not self.using_non_gui_backend:
            plt.pause(0.05)

