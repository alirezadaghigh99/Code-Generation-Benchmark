def Trace_ELBO(**kwargs):
    return elbo

class Adam:
    def __init__(self, optim_args):
        self.optim_args = optim_args
        # Each parameter will get its own optimizer, which we keep track
        # of using this dictionary keyed on parameters.
        self.optim_objs = {}

    def __call__(self, params):
        for param in params:
            # If we've seen this parameter before, use the previously
            # constructed optimizer.
            if param in self.optim_objs:
                optim = self.optim_objs[param]
            # If we've never seen this parameter before, construct
            # an Adam optimizer and keep track of it.
            else:
                optim = torch.optim.Adam([param], **self.optim_args)
                self.optim_objs[param] = optim
            # Take a gradient step for the parameter param.
            optim.step()

