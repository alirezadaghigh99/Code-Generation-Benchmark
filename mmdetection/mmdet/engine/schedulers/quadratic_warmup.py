class QuadraticWarmupMomentum(MomentumSchedulerMixin,
                              QuadraticWarmupParamScheduler):
    """Warm up the momentum value of each parameter group by quadratic formula.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """