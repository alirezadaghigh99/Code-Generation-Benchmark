def functional_call(*args, **kwargs):
            with stateless._reparametrize_module(foo, {}):
                return foo(*args, **kwargs)

