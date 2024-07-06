def transpose(t, batch_ndim):
            if layout is torch.sparse_csc:
                return t.transpose(batch_ndim, batch_ndim + 1)
            return t

def numel(tensor):
            r = 1
            for s in tensor.shape:
                r *= s
            return r

