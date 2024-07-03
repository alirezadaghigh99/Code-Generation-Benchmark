        def randint():
            return torch.randint(0, 5, [1, 2])        def add(a, b):
            return a + b            def addmm(mat, mat1, mat2):
                a = mat.addmm(mat1, mat2)
                b = mat.addmm(mat1, mat2, alpha=1.0, beta=1.0)
                return a + b        class D(C, B):
            def __init__(self):
                super().__init__()        def add(a, b):
            return a + b        def stft(input, n_fft):
            # type: (Tensor, int) -> Tensor
            return torch.stft(input, n_fft, return_complex=True)            def addmm(mat, mat1, mat2):
                a = mat.addmm(mat1, mat2)
                b = mat.addmm(mat1, mat2, alpha=1.0, beta=1.0)
                return a + b