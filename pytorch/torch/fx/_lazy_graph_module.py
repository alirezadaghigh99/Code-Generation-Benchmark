def _force_skip_lazy_graph_module():
    """
    Skip using lazy graph module disregarding the setting of _use_lazy_graph_module.
    Use to skip _LazyGraphModule when testing inductor torchscript related backend.

    torch.jit.script a _LazyGraphModule results in following error:
        https://gist.github.com/shunting314/5143654c8084aed84ecd19b818258a69
    """
    try:
        global _force_skip_lazy_graph_module_flag
        prior = _force_skip_lazy_graph_module_flag
        _force_skip_lazy_graph_module_flag = True
        yield
    finally:
        _force_skip_lazy_graph_module_flag = prior

