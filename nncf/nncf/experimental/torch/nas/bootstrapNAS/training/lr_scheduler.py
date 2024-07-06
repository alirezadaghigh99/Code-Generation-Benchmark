def calc_learning_rate(
    epoch: float,
    init_lr: float,
    n_epochs: float,
    batch: float = 0,
    n_batch: float = 0,
    lr_schedule_type: str = "cosine",
):
    if lr_schedule_type == "cosine":
        t_total = n_epochs * n_batch
        t_cur = epoch * n_batch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError("do not support: %s" % lr_schedule_type)
    return lr

