def clamp(self, data, min, max):
        return torch.clamp(data, min=min, max=max)

def rand_like(self, v):
        return torch.rand_like(v)

def batch_norm(self, data, mean, var, training):
        return torch.nn.functional.batch_norm(data, mean, var, training=training)

def matmul(self, t1, t2):
        return torch.matmul(t1, t2)

def matmul(self, t1, t2):
        return torch.matmul(t1, t2)

def softmax(self, data, dim=None, dtype=None):
        return torch.nn.functional.softmax(data, dim, dtype)

