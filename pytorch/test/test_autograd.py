def where(cond, x, y):
            return torch.where(cond, x, y)

def jvp(tangent):
            with fwAD.dual_level():
                x = fwAD.make_dual(input, tangent)
                return fwAD.unpack_dual(x)[1]

def sum(fn):
        def wrapped(x):
            return fn(x).sum()

        return wrapped

def sum(fn):
        def wrapped(x):
            return fn(x).sum()

        return wrapped

def grad(fn):
        def wrapper(x):
            with torch.enable_grad():
                out = fn(x)
                (grad_input,) = torch.autograd.grad(out, inputs=(x,), create_graph=True)
            return grad_input

        return wrapper

def grad(fn):
        def wrapper(x):
            with torch.enable_grad():
                out = fn(x)
                (grad_input,) = torch.autograd.grad(out, inputs=(x,), create_graph=True)
            return grad_input

        return wrapper

def foo(t1, t2, t3):
            res = MyFunction.apply(t1, t2, scale, t3)
            return res[1], res[4], res[6]

def vjp(ctx, gO):
                if ctx.bwd_bad:
                    raise RuntimeError(bwd_fail_err_msg)
                else:
                    return 2 * gO, None, None

def check(fast_mode):
            def f(inp):
                return inp.mul(5)

            gradcheck(
                f,
                torch.rand(10, dtype=torch.float64, requires_grad=True),
                fast_mode=fast_mode,
            )
            gradgradcheck(
                f,
                torch.rand(10, dtype=torch.float64, requires_grad=True),
                fast_mode=fast_mode,
            )

