def create_optim_sgd(model: Module, lr: float = 0.0001) -> SGD:
    return SGD(model.parameters(), lr=lr)

