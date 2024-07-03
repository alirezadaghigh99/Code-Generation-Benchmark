def _hessian_vector_product(flat_grads, params, vec):
    """Compute hessian vector product efficiently by backprop."""
    vec = vec.detach()
    grads = torch.autograd.grad(
        [torch.sum(flat_grads * vec)], params, retain_graph=True
    )
    assert all(
        grad is not None for grad in grads
    ), "The Hessian-vector product contains None."
    return _flatten_and_concat_variables(grads)