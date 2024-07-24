def randint():
            return torch.randint(0, 5, [1, 2])

def add(a, b):
            return a + b

def addmm(mat, mat1, mat2):
                a = mat.addmm(mat1, mat2)
                b = mat.addmm(mat1, mat2, alpha=1.0, beta=1.0)
                return a + b

def add(a, b):
            return a + b

def stft(input, n_fft):
            # type: (Tensor, int) -> Tensor
            return torch.stft(input, n_fft, return_complex=True)

def addmm(mat, mat1, mat2):
                a = mat.addmm(mat1, mat2)
                b = mat.addmm(mat1, mat2, alpha=1.0, beta=1.0)
                return a + b

def profile(func, X):
            with torch.autograd.profiler.profile() as prof:
                func(X)
            return [e.name for e in prof.function_events]

def enable_grad():
                    torch.set_grad_enabled(True)

def mul(a, x):
            return a * x

def t(x):
            gather1 = x[0]
            idx = 0 + 1
            gather2 = x[idx]
            return gather1 + gather2

def istft(input, n_fft):
            # type: (Tensor, int) -> Tensor
            return torch.istft(input, n_fft)

def norm():
            c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)
            return torch.norm(c, p="fro"), torch.norm(c, p="nuc"), torch.norm(c), torch.norm(c, p=.5)

def broadcast(a, b):
            return a + b

class D(C, B):
            def __init__(self):
                super().__init__()

