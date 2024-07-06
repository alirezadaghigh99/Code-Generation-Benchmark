def get_adv_criterion(loss_type: str) -> tp.Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == 'mse':
        return mse_loss
    elif loss_type == 'hinge':
        return hinge_loss
    elif loss_type == 'hinge2':
        return hinge2_loss
    raise ValueError('Unsupported loss')

