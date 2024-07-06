def transpose(t, batch_ndim):
            if layout is torch.sparse_csc:
                return t.transpose(batch_ndim, batch_ndim + 1)
            return t

